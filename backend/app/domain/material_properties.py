"""FEA material property database for ultrasonic welding horn/tool materials.

Each entry contains mechanical and thermal properties required for
modal analysis, harmonic response, and thermal-structural coupling.
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Material property database
# ---------------------------------------------------------------------------
# Keys:
#   E_pa            -- Young's modulus [Pa]
#   nu              -- Poisson's ratio  [-]
#   rho_kg_m3       -- Density [kg/m^3]
#   k_w_mk          -- Thermal conductivity [W/(m*K)]
#   cp_j_kgk        -- Specific heat capacity [J/(kg*K)]
#   yield_mpa       -- Yield strength [MPa]
#   alpha_1_k       -- Coefficient of thermal expansion [1/K]
# ---------------------------------------------------------------------------

FEA_MATERIALS: dict[str, dict] = {
    "Titanium Ti-6Al-4V": {
        "E_pa": 113.8e9,
        "nu": 0.342,
        "rho_kg_m3": 4430.0,
        "k_w_mk": 6.7,
        "cp_j_kgk": 526.3,
        "yield_mpa": 880.0,
        "alpha_1_k": 8.6e-6,
    },
    "Steel D2": {
        "E_pa": 210.0e9,
        "nu": 0.30,
        "rho_kg_m3": 7700.0,
        "k_w_mk": 20.0,
        "cp_j_kgk": 460.0,
        "yield_mpa": 1620.0,
        "alpha_1_k": 10.4e-6,
    },
    "Aluminum 7075-T6": {
        "E_pa": 71.7e9,
        "nu": 0.33,
        "rho_kg_m3": 2810.0,
        "k_w_mk": 130.0,
        "cp_j_kgk": 960.0,
        "yield_mpa": 503.0,
        "alpha_1_k": 23.6e-6,
    },
    "Copper C11000": {
        "E_pa": 117.0e9,
        "nu": 0.34,
        "rho_kg_m3": 8940.0,
        "k_w_mk": 388.0,
        "cp_j_kgk": 385.0,
        "yield_mpa": 69.0,
        "alpha_1_k": 17.0e-6,
    },
    "Nickel 200": {
        "E_pa": 204.0e9,
        "nu": 0.31,
        "rho_kg_m3": 8890.0,
        "k_w_mk": 70.2,
        "cp_j_kgk": 456.0,
        "yield_mpa": 148.0,
        "alpha_1_k": 13.3e-6,
    },
    "M2 High Speed Steel": {
        "E_pa": 220.0e9,      # Young's modulus 220 GPa
        "nu": 0.28,            # Poisson's ratio
        "rho_kg_m3": 8160.0,   # Density
        "k_w_mk": 25.9,        # Thermal conductivity
        "cp_j_kgk": 420.0,     # Specific heat
        "yield_mpa": 2200.0,   # Yield strength (hardened)
        "alpha_1_k": 11.0e-6,  # CTE
    },
    "CPM 10V": {
        "E_pa": 222.0e9,       # Young's modulus 222 GPa
        "nu": 0.29,            # Poisson's ratio
        "rho_kg_m3": 7690.0,   # Density
        "k_w_mk": 20.4,        # Thermal conductivity
        "cp_j_kgk": 430.0,     # Specific heat
        "yield_mpa": 2100.0,   # Yield strength (hardened)
        "alpha_1_k": 10.8e-6,  # CTE
    },
    "PM60 Powder Steel": {
        "E_pa": 230.0e9,       # Young's modulus 230 GPa
        "nu": 0.28,            # Poisson's ratio
        "rho_kg_m3": 8100.0,   # Density
        "k_w_mk": 24.0,        # Thermal conductivity
        "cp_j_kgk": 430.0,     # Specific heat
        "yield_mpa": 2400.0,   # Yield strength
        "alpha_1_k": 10.5e-6,  # CTE
    },
    "HAP40 Powder HSS": {
        "E_pa": 228.0e9,       # Young's modulus 228 GPa
        "nu": 0.28,            # Poisson's ratio
        "rho_kg_m3": 8050.0,   # Density
        "k_w_mk": 23.0,        # Thermal conductivity
        "cp_j_kgk": 425.0,     # Specific heat
        "yield_mpa": 2500.0,   # Yield strength
        "alpha_1_k": 10.7e-6,  # CTE
    },
    "HAP72 Powder HSS": {
        "E_pa": 235.0e9,       # Young's modulus 235 GPa
        "nu": 0.27,            # Poisson's ratio
        "rho_kg_m3": 8200.0,   # Density
        "k_w_mk": 21.0,        # Thermal conductivity
        "cp_j_kgk": 420.0,     # Specific heat
        "yield_mpa": 2800.0,   # Yield strength
        "alpha_1_k": 10.3e-6,  # CTE
    },
}

# ---------------------------------------------------------------------------
# Alias mapping for case-insensitive and shorthand lookups
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    # Titanium aliases
    "titanium": "Titanium Ti-6Al-4V",
    "titanium ti-6al-4v": "Titanium Ti-6Al-4V",
    "ti-6al-4v": "Titanium Ti-6Al-4V",
    "ti6al4v": "Titanium Ti-6Al-4V",
    "ti64": "Titanium Ti-6Al-4V",
    # Steel D2 aliases
    "steel d2": "Steel D2",
    "d2": "Steel D2",
    "d2 steel": "Steel D2",
    "tool steel": "Steel D2",
    # Aluminum aliases
    "aluminum 7075-t6": "Aluminum 7075-T6",
    "aluminum 7075": "Aluminum 7075-T6",
    "al 7075-t6": "Aluminum 7075-T6",
    "al7075": "Aluminum 7075-T6",
    "7075-t6": "Aluminum 7075-T6",
    "7075": "Aluminum 7075-T6",
    "aluminum": "Aluminum 7075-T6",
    # Copper aliases
    "copper c11000": "Copper C11000",
    "copper": "Copper C11000",
    "c11000": "Copper C11000",
    "cu": "Copper C11000",
    # Nickel aliases
    "nickel 200": "Nickel 200",
    "nickel": "Nickel 200",
    "ni200": "Nickel 200",
    "ni 200": "Nickel 200",
    # M2 High Speed Steel aliases
    "m2 high speed steel": "M2 High Speed Steel",
    "m2": "M2 High Speed Steel",
    "m2 hss": "M2 High Speed Steel",
    "aisi m2": "M2 High Speed Steel",
    # CPM 10V aliases
    "cpm 10v": "CPM 10V",
    "cpm10v": "CPM 10V",
    "10v": "CPM 10V",
    # PM60 Powder Steel aliases
    "pm60 powder steel": "PM60 Powder Steel",
    "pm60": "PM60 Powder Steel",
    # HAP40 Powder HSS aliases
    "hap40 powder hss": "HAP40 Powder HSS",
    "hap40": "HAP40 Powder HSS",
    # HAP72 Powder HSS aliases
    "hap72 powder hss": "HAP72 Powder HSS",
    "hap72": "HAP72 Powder HSS",
}


def get_material(name: str) -> Optional[dict]:
    """Look up material properties by name (case-insensitive, alias-aware).

    Parameters
    ----------
    name:
        Material name or alias, e.g. ``"Titanium"``, ``"ti64"``, ``"D2"``.

    Returns
    -------
    dict or None
        A **copy** of the property dictionary, or ``None`` if not found.
    """
    # Try exact match first
    if name in FEA_MATERIALS:
        return dict(FEA_MATERIALS[name])

    # Try case-insensitive alias lookup
    key = _ALIASES.get(name.lower().strip())
    if key is not None:
        return dict(FEA_MATERIALS[key])

    return None


def list_materials() -> list[str]:
    """Return a sorted list of canonical material names."""
    return sorted(FEA_MATERIALS.keys())
