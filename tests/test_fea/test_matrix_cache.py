"""Tests for K/M matrix caching in GlobalAssembler."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import (
    GlobalAssembler,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_mesh() -> FEAMesh:
    """A coarse TET10 cylinder mesh for cache tests."""
    mesher = GmshMesher()
    return mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
        mesh_size=8.0,
        order=2,
    )


@pytest.fixture(scope="module")
def alternate_mesh() -> FEAMesh:
    """A different coarse TET10 mesh (different dimensions) for testing
    that cache keys change when the mesh changes."""
    mesher = GmshMesher()
    return mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={"diameter_mm": 30.0, "length_mm": 60.0},
        mesh_size=8.0,
        order=2,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure cache is empty before and after every test."""
    GlobalAssembler.clear_cache()
    yield
    GlobalAssembler.clear_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMatrixCache:
    """Tests for the class-level K/M matrix cache."""

    def test_cache_miss_then_hit(self, coarse_mesh: FEAMesh):
        """First call is a cache miss (assembles), second is a cache hit."""
        assembler = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")

        # First call: cache miss -> store
        K1, M1 = assembler.assemble()
        assert len(GlobalAssembler._cache) == 1

        # Second call: cache hit
        K2, M2 = assembler.assemble()
        assert len(GlobalAssembler._cache) == 1  # no new entry

        # Should return the exact same objects (not copies)
        assert K1 is K2
        assert M1 is M2

    def test_cache_key_changes_with_material(self, coarse_mesh: FEAMesh):
        """Different materials on the same mesh produce different cache keys."""
        a1 = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")
        a2 = GlobalAssembler(coarse_mesh, "Aluminum 7075-T6")

        key1 = a1._cache_key()
        key2 = a2._cache_key()

        assert key1 != key2, (
            "Cache keys should differ when material changes"
        )

    def test_cache_key_changes_with_mesh(
        self, coarse_mesh: FEAMesh, alternate_mesh: FEAMesh
    ):
        """Different meshes with the same material produce different cache keys."""
        a1 = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")
        a2 = GlobalAssembler(alternate_mesh, "Titanium Ti-6Al-4V")

        key1 = a1._cache_key()
        key2 = a2._cache_key()

        assert key1 != key2, (
            "Cache keys should differ when mesh geometry changes"
        )

    def test_use_cache_false_bypasses_cache(self, coarse_mesh: FEAMesh):
        """use_cache=False should not read from or write to the cache."""
        assembler = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")

        K1, M1 = assembler.assemble(use_cache=False)
        # Cache should remain empty
        assert len(GlobalAssembler._cache) == 0

        # Pre-populate cache
        K2, M2 = assembler.assemble(use_cache=True)
        assert len(GlobalAssembler._cache) == 1

        # use_cache=False should re-assemble even though cache has data
        K3, M3 = assembler.assemble(use_cache=False)
        # K3 should be a freshly assembled matrix, not the same object
        assert K3 is not K2
        assert M3 is not M2

    def test_clear_cache(self, coarse_mesh: FEAMesh):
        """clear_cache() should empty the cache."""
        assembler = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")
        assembler.assemble()
        assert len(GlobalAssembler._cache) == 1

        GlobalAssembler.clear_cache()
        assert len(GlobalAssembler._cache) == 0

    def test_cache_eviction_at_max_entries(self, coarse_mesh: FEAMesh):
        """When cache reaches max_entries, oldest entry should be evicted."""
        # Save original max and set to a small value for testing
        original_max = GlobalAssembler._cache_max_entries
        GlobalAssembler._cache_max_entries = 2

        try:
            # Populate cache with 2 entries using different materials
            a1 = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")
            a2 = GlobalAssembler(coarse_mesh, "Aluminum 7075-T6")

            a1.assemble()
            key1 = a1._cache_key()
            assert len(GlobalAssembler._cache) == 1

            a2.assemble()
            key2 = a2._cache_key()
            assert len(GlobalAssembler._cache) == 2

            # Now add a third entry -- should evict the first (oldest)
            a3 = GlobalAssembler(coarse_mesh, "Steel D2")
            a3.assemble()
            key3 = a3._cache_key()

            assert len(GlobalAssembler._cache) == 2
            assert key1 not in GlobalAssembler._cache, (
                "Oldest entry should have been evicted"
            )
            assert key2 in GlobalAssembler._cache
            assert key3 in GlobalAssembler._cache
        finally:
            GlobalAssembler._cache_max_entries = original_max

    def test_cached_matrices_identical_to_fresh(self, coarse_mesh: FEAMesh):
        """Cached matrices must be numerically identical to fresh assembly."""
        assembler = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")

        # First call: cache miss -> assembles and stores
        K_cached, M_cached = assembler.assemble(use_cache=True)

        # Force a fresh assembly bypassing cache
        K_fresh, M_fresh = assembler.assemble(use_cache=False)

        # Verify numerical equality
        K_diff = K_cached - K_fresh
        M_diff = M_cached - M_fresh

        assert K_diff.nnz == 0 or np.max(np.abs(K_diff.data)) == 0.0, (
            "Cached K differs from freshly assembled K"
        )
        assert M_diff.nnz == 0 or np.max(np.abs(M_diff.data)) == 0.0, (
            "Cached M differs from freshly assembled M"
        )

    def test_cache_shared_across_instances(self, coarse_mesh: FEAMesh):
        """Cache is class-level, so different instances share it."""
        a1 = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")
        a2 = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")

        K1, M1 = a1.assemble()
        K2, M2 = a2.assemble()

        # Second instance should get the cached result from the first
        assert K1 is K2
        assert M1 is M2
        assert len(GlobalAssembler._cache) == 1

    def test_cache_key_is_deterministic(self, coarse_mesh: FEAMesh):
        """Same mesh + material should always produce the same cache key."""
        a1 = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")
        a2 = GlobalAssembler(coarse_mesh, "Titanium Ti-6Al-4V")

        assert a1._cache_key() == a2._cache_key()
