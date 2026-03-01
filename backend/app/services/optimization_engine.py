"""Optimization engine with parametric sweep, Bayesian, and genetic algorithm strategies.

All numerical operations use numpy only (no sklearn, scipy.optimize, or pymoo).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from backend.app.schemas.optimization import Constraint, DesignVariable, Objective

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _decode_variables(variables: list[DesignVariable] | list[dict]) -> list[DesignVariable]:
    """Ensure variables are DesignVariable instances (may come from JSON)."""
    out: list[DesignVariable] = []
    for v in variables:
        if isinstance(v, dict):
            out.append(DesignVariable(**v))
        else:
            out.append(v)
    return out


def _decode_objectives(objectives: list[Objective] | list[dict]) -> list[Objective]:
    """Ensure objectives are Objective instances."""
    out: list[Objective] = []
    for o in objectives:
        if isinstance(o, dict):
            out.append(Objective(**o))
        else:
            out.append(o)
    return out


def _decode_constraints(constraints: list[Constraint] | list[dict] | None) -> list[Constraint]:
    """Ensure constraints are Constraint instances."""
    if not constraints:
        return []
    out: list[Constraint] = []
    for c in constraints:
        if isinstance(c, dict):
            out.append(Constraint(**c))
        else:
            out.append(c)
    return out


def _continuous_vars(variables: list[DesignVariable]) -> list[DesignVariable]:
    """Return only continuous variables."""
    return [v for v in variables if v.var_type == "continuous"]


def _categorical_vars(variables: list[DesignVariable]) -> list[DesignVariable]:
    """Return only categorical variables."""
    return [v for v in variables if v.var_type == "categorical"]


def _clip_to_bounds(value: float, var: DesignVariable) -> float:
    """Clip a value to the variable bounds, respecting step size."""
    lo = var.min_value if var.min_value is not None else -1e12
    hi = var.max_value if var.max_value is not None else 1e12
    value = float(np.clip(value, lo, hi))
    if var.step is not None and var.step > 0:
        value = lo + round((value - lo) / var.step) * var.step
        value = float(np.clip(value, lo, hi))
    return value


# ============================================================================
# Strategy 1: Parametric Sweep (Latin Hypercube Sampling)
# ============================================================================


class ParametricSweepStrategy:
    """Generate parameter combinations using Latin Hypercube Sampling."""

    def generate_samples(
        self,
        variables: list[DesignVariable],
        n_samples: int,
    ) -> list[dict[str, Any]]:
        """Generate *n_samples* parameter sets using LHS.

        For continuous variables the unit interval [0, 1] is divided into
        *n_samples* strata; one random sample is drawn from each stratum and
        the columns are independently permuted.  The result is then mapped to
        ``[min_value, max_value]``.

        For categorical variables the choices are cycled (round-robin) so that
        every category appears roughly equally often.
        """
        variables = _decode_variables(variables)
        cont = _continuous_vars(variables)
        cat = _categorical_vars(variables)

        # --- LHS for continuous variables ---
        n_cont = len(cont)
        if n_cont > 0 and n_samples > 0:
            # Create stratified samples: each column independently permuted
            rng = np.random.default_rng()
            lhs_matrix = np.zeros((n_samples, n_cont))
            for j in range(n_cont):
                perm = rng.permutation(n_samples)
                for i in range(n_samples):
                    lhs_matrix[i, j] = (perm[i] + rng.random()) / n_samples
        else:
            lhs_matrix = np.empty((n_samples, 0))

        # --- Build parameter dicts ---
        samples: list[dict[str, Any]] = []
        for i in range(n_samples):
            params: dict[str, Any] = {}

            # Continuous
            for j, var in enumerate(cont):
                lo = var.min_value if var.min_value is not None else 0.0
                hi = var.max_value if var.max_value is not None else 1.0
                raw = lo + lhs_matrix[i, j] * (hi - lo)
                params[var.name] = _clip_to_bounds(raw, var)

            # Categorical – cycle through choices
            for var in cat:
                choices = var.choices or []
                if choices:
                    params[var.name] = choices[i % len(choices)]

            samples.append(params)

        return samples

    def suggest_next(
        self,
        variables: list[DesignVariable],
        objectives: list[Objective],
        history: list[dict],
    ) -> dict[str, Any] | None:
        """Parametric sweep does not do iterative suggestions; return None."""
        return None


# ============================================================================
# Strategy 2: Bayesian Optimization (GP surrogate + Expected Improvement)
# ============================================================================


class BayesianStrategy:
    """Bayesian optimization using a Gaussian Process surrogate model.

    The GP implementation uses a squared-exponential (RBF) kernel with numpy
    matrix operations.  The acquisition function is Expected Improvement (EI).
    """

    def __init__(self) -> None:
        self.X_observed: list[np.ndarray] = []
        self.Y_observed: list[float] = []
        # GP hyper-parameters (length-scale and noise variance)
        self._length_scale: float = 1.0
        self._noise_var: float = 1e-6
        self._signal_var: float = 1.0

    # ----- GP kernel & prediction -------------------------------------

    @staticmethod
    def _rbf_kernel(
        X1: np.ndarray,
        X2: np.ndarray,
        length_scale: float,
        signal_var: float,
    ) -> np.ndarray:
        """Squared-exponential (RBF) kernel matrix."""
        # X1: (n, d), X2: (m, d)  -> K: (n, m)
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(
            X2**2, axis=1
        ) - 2 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0.0)
        return signal_var * np.exp(-0.5 * sq_dist / (length_scale**2))

    def _gp_predict(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at *X_test*.

        Returns (mu, sigma) each of shape (n_test,).
        """
        ls = self._length_scale
        sv = self._signal_var
        nv = self._noise_var

        K = self._rbf_kernel(X_train, X_train, ls, sv) + nv * np.eye(len(X_train))
        K_s = self._rbf_kernel(X_train, X_test, ls, sv)
        K_ss = self._rbf_kernel(X_test, X_test, ls, sv)

        # Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter
            K += 1e-4 * np.eye(len(K))
            L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
        mu = K_s.T @ alpha

        v = np.linalg.solve(L, K_s)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)
        sigma = np.sqrt(var)

        return mu, sigma

    # ----- Acquisition function: Expected Improvement -----------------

    @staticmethod
    def _expected_improvement(
        mu: np.ndarray,
        sigma: np.ndarray,
        y_best: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """Compute Expected Improvement (for *minimization*).

        EI(x) = (y_best - mu(x) - xi) * Phi(Z) + sigma(x) * phi(Z)
        where Z = (y_best - mu(x) - xi) / sigma(x)
        """
        improvement = y_best - mu - xi
        Z = improvement / sigma

        # Standard normal CDF and PDF via numpy (no scipy)
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        from numpy import sqrt as _sqrt

        Phi = 0.5 * (1.0 + _erf_approx(Z / _sqrt(2.0)))
        phi = np.exp(-0.5 * Z**2) / _sqrt(2.0 * np.pi)

        ei = improvement * Phi + sigma * phi
        # Where sigma ~ 0, EI = 0
        ei[sigma < 1e-12] = 0.0
        return ei

    # ----- Auto-tune length scale -------------------------------------

    def _auto_tune_length_scale(self, X: np.ndarray) -> None:
        """Set length_scale to median pairwise distance (heuristic)."""
        if len(X) < 2:
            return
        dists = np.sqrt(
            np.sum(X[:, None, :] - X[None, :, :], axis=-1) ** 2
            + 1e-30  # numerical guard
        )
        # Only upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(len(X), k=1)
        pairwise = dists[triu_idx]
        if len(pairwise) > 0:
            median_d = float(np.median(pairwise))
            if median_d > 0:
                self._length_scale = median_d

    # ----- Main interface ---------------------------------------------

    def generate_samples(
        self,
        variables: list[DesignVariable],
        n_samples: int,
    ) -> list[dict[str, Any]]:
        """Generate initial random samples (LHS) before Bayesian steps."""
        sweep = ParametricSweepStrategy()
        return sweep.generate_samples(variables, n_samples)

    def suggest_next(
        self,
        variables: list[DesignVariable],
        objectives: list[Objective],
        history: list[dict],
    ) -> dict[str, Any] | None:
        """Suggest the next parameter set by maximising Expected Improvement.

        For multi-objective problems, a scalar is formed as a weighted sum of
        the objectives (all normalised to *minimisation* direction).
        """
        variables = _decode_variables(variables)
        objectives = _decode_objectives(objectives)
        cont_vars = _continuous_vars(variables)

        if not cont_vars:
            return None
        if len(history) < 2:
            # Need at least 2 observations to fit GP
            sweep = ParametricSweepStrategy()
            samples = sweep.generate_samples(variables, 1)
            return samples[0] if samples else None

        # Build observed arrays from history
        X_list: list[list[float]] = []
        Y_list: list[float] = []

        for entry in history:
            params = entry.get("parameters", {})
            metrics = entry.get("metrics", {})
            if not metrics:
                continue

            row = []
            for var in cont_vars:
                row.append(float(params.get(var.name, 0.0)))
            X_list.append(row)

            # Scalarise objectives (weighted sum, flip sign for maximise)
            scalar = 0.0
            for obj in objectives:
                val = float(metrics.get(obj.metric, 0.0))
                if obj.direction == "maximize":
                    val = -val
                scalar += obj.weight * val
            Y_list.append(scalar)

        if len(X_list) < 2:
            sweep = ParametricSweepStrategy()
            samples = sweep.generate_samples(variables, 1)
            return samples[0] if samples else None

        X_obs = np.array(X_list)
        Y_obs = np.array(Y_list)

        # Normalise X to [0, 1]
        bounds_lo = np.array([v.min_value if v.min_value is not None else 0.0 for v in cont_vars])
        bounds_hi = np.array([v.max_value if v.max_value is not None else 1.0 for v in cont_vars])
        span = bounds_hi - bounds_lo
        span[span == 0] = 1.0
        X_norm = (X_obs - bounds_lo) / span

        # Normalise Y
        y_mean = Y_obs.mean()
        y_std = Y_obs.std()
        if y_std < 1e-12:
            y_std = 1.0
        Y_norm = (Y_obs - y_mean) / y_std

        self._auto_tune_length_scale(X_norm)

        # Generate candidate points (random + LHS)
        n_candidates = max(500, 50 * len(cont_vars))
        rng = np.random.default_rng()
        X_cand = rng.random((n_candidates, len(cont_vars)))

        mu, sigma = self._gp_predict(X_norm, Y_norm, X_cand)
        y_best = Y_norm.min()
        ei = self._expected_improvement(mu, sigma, y_best)

        best_idx = int(np.argmax(ei))
        x_best_norm = X_cand[best_idx]

        # Map back to original space
        x_best = bounds_lo + x_best_norm * span

        params: dict[str, Any] = {}
        for j, var in enumerate(cont_vars):
            params[var.name] = _clip_to_bounds(float(x_best[j]), var)

        # Handle categorical variables (pick random choice)
        for var in _categorical_vars(variables):
            choices = var.choices or []
            if choices:
                params[var.name] = choices[rng.integers(len(choices))]

        return params


# ============================================================================
# Strategy 3: Genetic Algorithm (NSGA-II style)
# ============================================================================


class GeneticStrategy:
    """NSGA-II inspired multi-objective genetic algorithm.

    Supports tournament selection, Simulated Binary Crossover (SBX) for
    continuous variables, polynomial mutation, and non-dominated sorting with
    crowding distance.
    """

    def __init__(
        self,
        population_size: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
    ) -> None:
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population: list[dict[str, Any]] = []
        self._rng = np.random.default_rng()

    # ----- Population initialisation -----------------------------------

    def initialize_population(
        self,
        variables: list[DesignVariable],
        pop_size: int,
    ) -> list[dict[str, Any]]:
        """Generate a random initial population."""
        variables = _decode_variables(variables)
        sweep = ParametricSweepStrategy()
        pop = sweep.generate_samples(variables, pop_size)
        self.population = pop
        return pop

    def generate_samples(
        self,
        variables: list[DesignVariable],
        n_samples: int,
    ) -> list[dict[str, Any]]:
        """Generate initial population (alias used by OptimizationEngine)."""
        self.population_size = n_samples
        return self.initialize_population(variables, n_samples)

    # ----- SBX crossover -----------------------------------------------

    def _sbx_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
        eta_c: float = 20.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover for continuous variables."""
        rng = self._rng
        c1 = p1.copy()
        c2 = p2.copy()

        for i in range(len(p1)):
            if rng.random() > 0.5:
                continue
            if abs(p1[i] - p2[i]) < 1e-14:
                continue

            lo = bounds_lo[i]
            hi = bounds_hi[i]

            if p1[i] > p2[i]:
                p1[i], p2[i] = p2[i], p1[i]

            # SBX spread factor
            u = rng.random()
            diff = p2[i] - p1[i]
            if diff < 1e-14:
                diff = 1e-14

            beta1 = 1.0 + 2.0 * (p1[i] - lo) / diff
            beta2 = 1.0 + 2.0 * (hi - p2[i]) / diff

            def _calc_beta_q(beta: float, u: float, eta: float) -> float:
                alpha_val = 2.0 - beta ** (-(eta + 1.0))
                if alpha_val < 1e-14:
                    alpha_val = 1e-14
                if u <= 1.0 / alpha_val:
                    return (u * alpha_val) ** (1.0 / (eta + 1.0))
                else:
                    return (1.0 / (2.0 - u * alpha_val)) ** (1.0 / (eta + 1.0))

            betaq1 = _calc_beta_q(beta1, u, eta_c)
            betaq2 = _calc_beta_q(beta2, u, eta_c)

            c1[i] = 0.5 * ((p1[i] + p2[i]) - betaq1 * diff)
            c2[i] = 0.5 * ((p1[i] + p2[i]) + betaq2 * diff)

            c1[i] = np.clip(c1[i], lo, hi)
            c2[i] = np.clip(c2[i], lo, hi)

        return c1, c2

    # ----- Polynomial mutation -----------------------------------------

    def _polynomial_mutation(
        self,
        x: np.ndarray,
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
        eta_m: float = 20.0,
    ) -> np.ndarray:
        """Polynomial mutation for continuous variables."""
        rng = self._rng
        y = x.copy()
        for i in range(len(x)):
            if rng.random() > self.mutation_rate:
                continue
            lo = bounds_lo[i]
            hi = bounds_hi[i]
            delta = hi - lo
            if delta < 1e-14:
                continue

            u = rng.random()
            if u < 0.5:
                delta_q = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))

            y[i] = x[i] + delta_q * delta
            y[i] = np.clip(y[i], lo, hi)

        return y

    # ----- Non-dominated sorting (NSGA-II) ----------------------------

    @staticmethod
    def _dominates(
        a: np.ndarray,
        b: np.ndarray,
    ) -> bool:
        """Return True if *a* dominates *b* (all objectives minimised)."""
        return bool(np.all(a <= b) and np.any(a < b))

    @staticmethod
    def non_dominated_sort(
        fitness: np.ndarray,
    ) -> list[list[int]]:
        """Fast non-dominated sorting.

        Parameters
        ----------
        fitness : np.ndarray of shape (n, m)
            Objective values (all to be *minimised*).

        Returns
        -------
        fronts : list of lists of indices, front 0 is the Pareto front.
        """
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)
        dominated_set: list[list[int]] = [[] for _ in range(n)]
        ranks = np.full(n, -1, dtype=int)
        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if GeneticStrategy._dominates(fitness[i], fitness[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif GeneticStrategy._dominates(fitness[j], fitness[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                ranks[i] = 0
                fronts[0].append(i)

        front_idx = 0
        while fronts[front_idx]:
            next_front: list[int] = []
            for i in fronts[front_idx]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)

        # Remove empty last front
        if not fronts[-1]:
            fronts.pop()

        return fronts

    @staticmethod
    def _crowding_distance(
        fitness: np.ndarray,
        front: list[int],
    ) -> np.ndarray:
        """Compute crowding distance for a single front."""
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        m = fitness.shape[1]

        for obj_idx in range(m):
            values = fitness[front, obj_idx]
            sorted_indices = np.argsort(values)
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            obj_range = values[sorted_indices[-1]] - values[sorted_indices[0]]
            if obj_range < 1e-14:
                continue

            for k in range(1, n - 1):
                distances[sorted_indices[k]] += (
                    values[sorted_indices[k + 1]] - values[sorted_indices[k - 1]]
                ) / obj_range

        return distances

    # ----- Tournament selection ----------------------------------------

    def _tournament_select(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        tournament_size: int = 2,
    ) -> int:
        """Binary tournament selection using rank then crowding distance."""
        rng = self._rng
        candidates = rng.integers(0, len(ranks), size=tournament_size)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and crowding[c] > crowding[best]:
                best = c
        return int(best)

    # ----- Evolve one generation --------------------------------------

    def evolve(
        self,
        population: list[dict[str, Any]],
        fitness: list[dict[str, float]],
        variables: list[DesignVariable],
    ) -> list[dict[str, Any]]:
        """Produce the next generation via selection, crossover, mutation.

        Parameters
        ----------
        population : current parameter dicts
        fitness : list of metric dicts per individual
        variables : design variable definitions

        Returns
        -------
        next_population : list of new parameter dicts
        """
        variables = _decode_variables(variables)
        cont_vars = _continuous_vars(variables)
        cat_vars = _categorical_vars(variables)
        n_pop = len(population)

        if n_pop == 0:
            return []

        # Build bounds arrays for continuous vars
        bounds_lo = np.array([v.min_value if v.min_value is not None else 0.0 for v in cont_vars])
        bounds_hi = np.array([v.max_value if v.max_value is not None else 1.0 for v in cont_vars])

        # Encode population as numpy arrays (continuous part only)
        X = np.zeros((n_pop, len(cont_vars)))
        for i, p in enumerate(population):
            for j, var in enumerate(cont_vars):
                X[i, j] = float(p.get(var.name, 0.0))

        # Build fitness matrix (use first objective if single)
        # Determine objectives from fitness keys
        if not fitness:
            return population

        # Just use raw metric values; direction handled externally
        # For NSGA-II we need all objective values
        obj_keys = list(fitness[0].keys()) if fitness[0] else []
        F = np.zeros((n_pop, max(len(obj_keys), 1)))
        for i, f in enumerate(fitness):
            for j, k in enumerate(obj_keys):
                F[i, j] = float(f.get(k, 0.0))

        # Non-dominated sorting
        fronts = self.non_dominated_sort(F)

        # Assign ranks and crowding distances
        ranks = np.zeros(n_pop, dtype=int)
        crowding = np.zeros(n_pop)
        for front_rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = front_rank
            cd = self._crowding_distance(F, front)
            for k, idx in enumerate(front):
                crowding[idx] = cd[k]

        # Generate offspring
        offspring: list[dict[str, Any]] = []
        while len(offspring) < n_pop:
            p1_idx = self._tournament_select(ranks, crowding)
            p2_idx = self._tournament_select(ranks, crowding)
            # Avoid same parent
            attempts = 0
            while p2_idx == p1_idx and attempts < 5:
                p2_idx = self._tournament_select(ranks, crowding)
                attempts += 1

            if self._rng.random() < self.crossover_rate and len(cont_vars) > 0:
                c1, c2 = self._sbx_crossover(
                    X[p1_idx], X[p2_idx], bounds_lo, bounds_hi
                )
            else:
                c1, c2 = X[p1_idx].copy(), X[p2_idx].copy()

            # Mutation
            if len(cont_vars) > 0:
                c1 = self._polynomial_mutation(c1, bounds_lo, bounds_hi)
                c2 = self._polynomial_mutation(c2, bounds_lo, bounds_hi)

            # Convert to dicts
            for child_arr in [c1, c2]:
                if len(offspring) >= n_pop:
                    break
                child: dict[str, Any] = {}
                for j, var in enumerate(cont_vars):
                    child[var.name] = _clip_to_bounds(float(child_arr[j]), var)
                # Categorical: inherit from random parent or mutate
                for var in cat_vars:
                    choices = var.choices or []
                    if choices:
                        if self._rng.random() < self.mutation_rate:
                            child[var.name] = choices[self._rng.integers(len(choices))]
                        else:
                            parent_params = population[p1_idx]
                            child[var.name] = parent_params.get(var.name, choices[0])
                offspring.append(child)

        return offspring[:n_pop]

    # ----- Pareto front -----------------------------------------------

    def compute_pareto_front(
        self,
        population: list[dict[str, Any]],
        fitness: list[dict[str, float]],
        objectives: list[Objective],
    ) -> list[int]:
        """Return indices of Pareto-optimal solutions."""
        objectives = _decode_objectives(objectives)
        n = len(population)
        if n == 0:
            return []

        # Build fitness matrix with correct direction
        F = np.zeros((n, len(objectives)))
        for i, f in enumerate(fitness):
            for j, obj in enumerate(objectives):
                val = float(f.get(obj.metric, 0.0))
                if obj.direction == "maximize":
                    val = -val  # Convert to minimisation
                F[i, j] = val

        fronts = self.non_dominated_sort(F)
        return fronts[0] if fronts else []

    def suggest_next(
        self,
        variables: list[DesignVariable],
        objectives: list[Objective],
        history: list[dict],
    ) -> dict[str, Any] | None:
        """For GA, evolve the population using available history.

        Collects the last *population_size* entries from history, evaluates
        their fitness, and returns the first individual of the evolved
        offspring.
        """
        variables = _decode_variables(variables)
        objectives = _decode_objectives(objectives)

        if not history:
            return None

        # Take the most recent population_size entries as the current population
        recent = history[-self.population_size:]
        pop = [entry.get("parameters", {}) for entry in recent]
        fitness_list: list[dict[str, float]] = []
        for entry in recent:
            metrics = entry.get("metrics", {})
            f: dict[str, float] = {}
            for obj in objectives:
                val = float(metrics.get(obj.metric, 0.0))
                if obj.direction == "maximize":
                    val = -val
                f[obj.metric] = val
            fitness_list.append(f)

        evolved = self.evolve(pop, fitness_list, variables)
        return evolved[0] if evolved else None


# ============================================================================
# Approximation of the error function (no scipy)
# ============================================================================

def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Approximation of erf(x) using Abramowitz & Stegun formula 7.1.26.

    Maximum error ~ 1.5e-7, good enough for EI calculations.
    """
    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x_abs = np.abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs**2)
    return sign * y


# ============================================================================
# Main engine class
# ============================================================================


class OptimizationEngine:
    """Orchestrates optimization studies with different strategies.

    Usage::

        engine = OptimizationEngine("bayesian")
        initial = engine.generate_initial_samples(variables, 10)
        # ... evaluate initial samples ...
        next_params = engine.suggest_next(variables, objectives, constraints, history)
    """

    STRATEGY_MAP = {
        "parametric_sweep": ParametricSweepStrategy,
        "bayesian": BayesianStrategy,
        "genetic": GeneticStrategy,
    }

    def __init__(self, strategy: str, **kwargs: Any) -> None:
        cls = self.STRATEGY_MAP.get(strategy)
        if cls is None:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Available: {list(self.STRATEGY_MAP)}"
            )
        self.strategy_name = strategy
        self.strategy = cls(**kwargs) if kwargs else cls()

    # ----- Generate initial samples -----------------------------------

    def generate_initial_samples(
        self,
        variables: list[DesignVariable] | list[dict],
        n_samples: int,
    ) -> list[dict[str, Any]]:
        """Generate *n_samples* initial parameter sets to evaluate."""
        variables = _decode_variables(variables)
        return self.strategy.generate_samples(variables, n_samples)

    # ----- Suggest next point -----------------------------------------

    def suggest_next(
        self,
        variables: list[DesignVariable] | list[dict],
        objectives: list[Objective] | list[dict],
        constraints: list[Constraint] | list[dict] | None,
        history: list[dict],
    ) -> dict[str, Any] | None:
        """Suggest the next parameter set, or None if converged / done.

        Also checks convergence: if the last *window* entries have objective
        improvement below *tol*, returns None.
        """
        variables = _decode_variables(variables)
        objectives = _decode_objectives(objectives)
        constraints = _decode_constraints(constraints)

        # Convergence check (for Bayesian and GA)
        if self.strategy_name in ("bayesian", "genetic") and len(history) >= 5:
            if self._check_convergence(history, objectives):
                logger.info("Optimization converged (objective stagnation).")
                return None

        result = self.strategy.suggest_next(variables, objectives, history)
        return result

    # ----- Convergence check ------------------------------------------

    @staticmethod
    def _check_convergence(
        history: list[dict],
        objectives: list[Objective],
        window: int = 5,
        tol: float = 1e-4,
    ) -> bool:
        """Return True if the best objective has not improved over *window* iterations."""
        if len(history) < window:
            return False

        recent = history[-window:]
        scalars: list[float] = []
        for entry in recent:
            metrics = entry.get("metrics", {})
            if not metrics:
                return False
            s = 0.0
            for obj in objectives:
                val = float(metrics.get(obj.metric, 0.0))
                if obj.direction == "maximize":
                    val = -val
                s += obj.weight * val
            scalars.append(s)

        if not scalars:
            return False

        rng = max(scalars) - min(scalars)
        scale = max(abs(scalars[0]), 1e-12)
        return rng / scale < tol

    # ----- Constraint evaluation --------------------------------------

    @staticmethod
    def evaluate_constraints(
        metrics: dict[str, float],
        constraints: list[Constraint] | list[dict] | None,
    ) -> bool:
        """Check if *metrics* satisfy all *constraints*.

        Supported operators: ``<=``, ``>=``, ``==``, ``within_percent``.
        """
        if not constraints:
            return True
        parsed = _decode_constraints(constraints)

        for c in parsed:
            actual = metrics.get(c.metric)
            if actual is None:
                # Metric not present - treat as infeasible
                return False
            actual = float(actual)
            target = float(c.value)

            if c.operator == "<=":
                if actual > target:
                    return False
            elif c.operator == ">=":
                if actual < target:
                    return False
            elif c.operator == "==":
                tol = c.tolerance_pct / 100.0 * abs(target) if c.tolerance_pct else 1e-9
                if abs(actual - target) > tol:
                    return False
            elif c.operator == "within_percent":
                pct = c.tolerance_pct if c.tolerance_pct is not None else c.value
                # Check if actual is within pct% of target
                # "within_percent" uses value as the reference and tolerance_pct as the %
                if c.tolerance_pct is not None:
                    ref = c.value
                    pct_val = c.tolerance_pct
                else:
                    # Fallback: value is the percentage, metric must be within value% of 0
                    ref = 0.0
                    pct_val = c.value
                threshold = abs(ref) * pct_val / 100.0 if ref != 0 else pct_val / 100.0
                if abs(actual - ref) > threshold:
                    return False
            else:
                logger.warning("Unknown constraint operator: %s", c.operator)

        return True

    # ----- Pareto front computation -----------------------------------

    def compute_pareto_front(
        self,
        history: list[dict],
        objectives: list[Objective] | list[dict],
    ) -> list[int]:
        """Compute Pareto front indices from iteration history.

        Parameters
        ----------
        history : list of iteration dicts with ``"metrics"`` keys
        objectives : objective definitions

        Returns
        -------
        List of indices into *history* that are Pareto-optimal.
        """
        objectives = _decode_objectives(objectives)
        n = len(history)
        if n == 0:
            return []

        # Build fitness matrix
        F = np.zeros((n, len(objectives)))
        for i, entry in enumerate(history):
            metrics = entry.get("metrics", {})
            for j, obj in enumerate(objectives):
                val = float(metrics.get(obj.metric, 0.0))
                if obj.direction == "maximize":
                    val = -val
                F[i, j] = val

        fronts = GeneticStrategy.non_dominated_sort(F)
        return fronts[0] if fronts else []
