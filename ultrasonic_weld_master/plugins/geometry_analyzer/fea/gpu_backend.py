"""GPU-accelerated eigensolver backend with automatic CPU fallback.

For the standard eigenvalue problem (A*x = lambda*x), uses CuPy's GPU eigsh.
For the generalized shift-invert problem used in FEA modal analysis
(K*x = lambda*M*x near sigma), uses SciPy's CPU ARPACK which is faster
than GPU for typical FEA sizes due to cuSOLVER sparse LU overhead.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU availability detection (module-level, runs once at import)
# ---------------------------------------------------------------------------
_GPU_AVAILABLE = False
_GPU_INFO: dict = {}

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_spla

    _GPU_AVAILABLE = True
    _dev = cp.cuda.Device(0)
    _mem_info = _dev.mem_info  # (free, total)
    _props = cp.cuda.runtime.getDeviceProperties(0)
    _GPU_INFO = {
        "name": _props["name"].decode() if isinstance(_props["name"], bytes) else str(_props["name"]),
        "vram_total_bytes": _mem_info[1],
        "vram_free_bytes": _mem_info[0],
        "cuda_version": cp.cuda.runtime.runtimeGetVersion(),
    }
    logger.info(
        "GPU backend available: %s (%.1f GB VRAM)",
        _GPU_INFO["name"],
        _GPU_INFO["vram_total_bytes"] / 1e9,
    )
except ImportError:
    logger.info("CuPy not installed; GPU eigensolver disabled, using CPU")
except Exception as exc:
    logger.warning("GPU detection failed: %s", exc)

# ---------------------------------------------------------------------------
# Concurrency control
# ---------------------------------------------------------------------------
_analysis_semaphore: asyncio.Semaphore | None = None


def get_analysis_semaphore(max_concurrent: int = 2) -> asyncio.Semaphore:
    """Return a shared semaphore limiting concurrent FEA analyses."""
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(max_concurrent)
    return _analysis_semaphore


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------
def _estimate_gpu_memory(K: sp.csr_matrix, k: int) -> int:
    """Estimate GPU VRAM needed in bytes for standard eigsh."""
    n_dof = K.shape[0]
    k_mem = K.nnz * 12 + (K.shape[0] + 1) * 4
    work_mem = n_dof * k * 8 * 4
    return int((k_mem + work_mem) * 1.3)


# ---------------------------------------------------------------------------
# Core eigsh wrapper
# ---------------------------------------------------------------------------
def gpu_eigsh(
    K: sp.csr_matrix,
    k: int,
    M: sp.csr_matrix | None = None,
    sigma: float | None = None,
    which: str = "LM",
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigsh with GPU acceleration where beneficial.

    For the standard eigenvalue problem (M=None, sigma=None), uses CuPy
    GPU eigsh.  For the generalized shift-invert problem (our primary FEA
    use case), SciPy CPU ARPACK is used directly — it is faster than GPU
    for typical FEA sizes (<200K DOFs) because cuSOLVER's sparse LU has
    higher overhead than SuperLU.

    Parameters match ``scipy.sparse.linalg.eigsh`` exactly.
    Always returns NumPy arrays for downstream compatibility.
    """
    # Generalized or shift-invert: CPU ARPACK is faster for FEA sizes
    if sigma is not None or M is not None:
        return _cpu_eigsh(K, k, M, sigma, which)

    # Standard eigenvalue problem: try GPU
    if use_gpu and _GPU_AVAILABLE:
        try:
            return _gpu_standard_eigsh(K, k, which)
        except Exception as exc:
            logger.warning("GPU eigsh failed (%s), falling back to CPU", exc)

    return _cpu_eigsh(K, k, M, sigma, which)


def _gpu_standard_eigsh(
    K: sp.csr_matrix,
    k: int,
    which: str,
) -> tuple[np.ndarray, np.ndarray]:
    """CuPy eigsh for standard eigenvalue problem (no M, no sigma)."""
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_spla

    n_dof = K.shape[0]
    mem_needed = _estimate_gpu_memory(K, k)
    free_mem = cp.cuda.Device(0).mem_info[0]
    if mem_needed > free_mem * 0.9:
        raise MemoryError(
            f"Insufficient GPU memory: need {mem_needed / 1e9:.2f} GB, "
            f"free {free_mem / 1e9:.2f} GB"
        )

    t0 = time.perf_counter()
    K_gpu = cp_sparse.csr_matrix(
        (cp.asarray(K.data), cp.asarray(K.indices), cp.asarray(K.indptr)),
        shape=K.shape,
    )
    eigenvalues_gpu, eigenvectors_gpu = cp_spla.eigsh(K_gpu, k=k, which=which)
    eigenvalues = cp.asnumpy(eigenvalues_gpu)
    eigenvectors = cp.asnumpy(eigenvectors_gpu)

    dt = time.perf_counter() - t0
    logger.info("GPU standard eigsh: %.2f s (%d DOFs, %d modes)", dt, n_dof, k)

    del K_gpu, eigenvalues_gpu, eigenvectors_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return eigenvalues, eigenvectors


def _cpu_eigsh(
    K: sp.csr_matrix,
    k: int,
    M: sp.csr_matrix | None,
    sigma: float | None,
    which: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard SciPy CPU eigsh."""
    t0 = time.perf_counter()
    kwargs: dict = {"k": k, "which": which}
    if M is not None:
        kwargs["M"] = M
    if sigma is not None:
        kwargs["sigma"] = sigma
    result = spla.eigsh(K, **kwargs)
    dt = time.perf_counter() - t0
    logger.info(
        "CPU eigsh: %.2f s (%d DOFs, k=%d, sigma=%s)",
        dt, K.shape[0], k, sigma,
    )
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def gpu_status() -> dict:
    """Return GPU backend status for health endpoints."""
    info = {"gpu_available": _GPU_AVAILABLE}
    if _GPU_AVAILABLE:
        import cupy as cp

        free, total = cp.cuda.Device(0).mem_info
        info.update({
            "gpu_name": _GPU_INFO.get("name", "unknown"),
            "vram_total_gb": round(total / 1e9, 1),
            "vram_free_gb": round(free / 1e9, 1),
        })
    return info
