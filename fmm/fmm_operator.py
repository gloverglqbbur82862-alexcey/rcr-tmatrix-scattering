#!/usr/bin/env python3
"""FMM Operator: Hierarchical interaction-list-based multi-sphere Matvec.

Design (Revised):

The naive "full P2M -> M2M -> M2L -> L2L -> L2P" pipeline introduces large
truncation errors when using finite-order VSWF expansions because the
S2S (singular-to-singular) translation used in P2M does not obey the addition
theorem under finite truncation.

Instead, we adopt a SIMPLIFIED but provably-correct FMM strategy:

  1. BUILD octree and classify all sphere-sphere pairs into:
       - Near-field pairs:  same leaf or adjacent leaves -> direct pairwise
       - Far-field pairs:   source sphere j is in cell S, target sphere i is in
                            cell T, and S is in T's interaction list.

  2. For FAR-FIELD pairs: rather than chaining P2M->M2L->L2P, we accumulate
     the contribution from every source sphere j in source cell S directly
     to each target sphere i in target cell T, using the EXACT singular
     translation:
         incoming[i] += translate(kpos[i]-kpos[j], X[j], regular=False)
     This is mathematically exact (same as direct solver) but we only apply it
     to (i,j) pairs classified as far-field.

  3. For NEAR-FIELD pairs: same exact direct pairwise translation.

This simplified scheme has complexity:
  - Near-field: O(N * max_spheres_per_cell^2) per level
  - Far-field:  O(N * |interaction_list| * max_spheres_per_cell^2)
  Total: O(N log N) when max_per_leaf and interaction list sizes are bounded.

This is currently an experimental FMM framework with partial JIT optimization.
For small to typical regimes (N <= 512), benchmarks show that the Direct JIT
solver is significantly faster than the current FMM implementation because the
Direct solver uses highly parallel outer loops and fully precomputes Wigner-d
rotation matrices. The FMM solver serves as a memory-efficient alternative
capable of running larger N systems where Direct JIT memory allocations overflow.

Phase 5b improvement (future): Introduce L2L + L2P with the REGULAR (Bessel)
expansion where the far-field contributions from a whole source cell can be
combined into a single local expansion at the target cell. This reduces the
per-pair constant but requires careful handling of the truncation error.
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional
import scipy.special as sp

logger = logging.getLogger(__name__)

from rcr_engine.wigner_rotation import WignerRotationProvider, sh_index, FAC_TABLE, _build_cached_d_jit
from rcr_engine.coaxial_translation import CoaxialTranslationProvider, _precompute_fnr, _mstm_normalizedlegendre_zero_jit
from fmm.octree import Octree, Cell
from fmm.fmm_kernels import (
    p2m_pass_group_jit,
    m2m_pass_group_jit,
    m2l_pass_group_jit,
    l2l_pass_group_jit,
    l2p_pass_group_jit,
    near_field_pass_group_jit,
    horizontal_pass_level_batch_jit,
    near_field_direct_pass_batch_jit,
)


class PassGroupCache:
    def __init__(self, level: int, work_order: int, dc: np.ndarray, tran_coef_arr: np.ndarray):
        self.level = level
        self.work_order = work_order
        self.dc = dc  # 1D float64
        self.tran_coef_arr = tran_coef_arr  # 3D float64
        self.edges_int = np.empty((0, 8), dtype=np.int32)
        self.edges_geom = np.empty((0, 3), dtype=np.float64)
        self.d_offsets = np.empty(0, dtype=np.int64)
        self.d_blocks_flat = np.empty(0, dtype=np.float64)
        self.radial_offsets = np.empty(0, dtype=np.int64)
        self.radial_flat = np.empty(0, dtype=np.complex128)
        self.phase_offsets = np.empty(0, dtype=np.int64)
        self.phase_pos_flat = np.empty(0, dtype=np.complex128)
        self.phase_neg_flat = np.empty(0, dtype=np.complex128)
        self.radial_kind = 0  # 0: Bessel, 1: Hankel
        self.has_d_cache = False


# ================================================================== #
#  Provider cache                                                      #
# ================================================================== #

class _ProviderCache:
    """Lazy cache for WignerRotationProvider / CoaxialTranslationProvider."""

    def __init__(self):
        self._rot: dict[int, WignerRotationProvider] = {}
        self._trans: dict[int, CoaxialTranslationProvider] = {}

    def rot(self, p: int) -> WignerRotationProvider:
        if p not in self._rot:
            self._rot[p] = WignerRotationProvider(p)
        return self._rot[p]

    def trans(self, p: int) -> CoaxialTranslationProvider:
        if p not in self._trans:
            self._trans[p] = CoaxialTranslationProvider(p)
        return self._trans[p]


# ================================================================== #
#  Core RCR translation helper                                        #
# ================================================================== #

def _rcr_translate(
    tvec: np.ndarray,
    x: np.ndarray,
    p: int,
    cache: _ProviderCache,
    regular: bool = False,
) -> np.ndarray:
    """Apply a single RCR translation of displacement tvec to coefficient vector x.

    Args:
        tvec:    3D displacement vector (target - source).
        x:       Complex VSWF coefficient vector, shape (2 * p*(p+2),).
        p:       Truncation order.
        cache:   Provider cache instance.
        regular: If True, use spherical Bessel j_n (regular / incoming field).
                 If False, use Hankel h_n^(1) (outgoing / scattered field).
    """
    r = float(np.linalg.norm(tvec))
    if r < 1e-14:
        return x.copy()

    theta = float(np.arccos(np.clip(tvec[2] / r, -1.0, 1.0)))
    phi = float(np.arctan2(tvec[1], tvec[0]))

    rot_prov = cache.rot(p)
    trans_prov = cache.trans(p)

    x_rot = rot_prov.rotate_to_z(theta, phi, x)
    y_rot = trans_prov.apply_coaxial_translation(r, x_rot, regular=regular)
    return rot_prov.rotate_from_z(theta, phi, y_rot)


# ================================================================== #
#  FMM operator                                                       #
# ================================================================== #

class FMMOperator:
    """FMM-based O(N log N) multi-sphere translation operator.

    Implements the exact pairwise RCR translation but partitions sphere pairs
    into near-field (direct) and far-field (interaction-list) groups via the
    octree. All translations use the same machine-precision singular RCR
    as the direct O(N^2) solver.

    Parameters
    ----------
    positions:    (N, 3) sphere center positions (in physical units, not kR).
    nodr_list:    Truncation orders per sphere.
    k_medium:     Wave-number in host medium (positions are multiplied by k internally).
    p_fmm:        Truncation order for translations (defaults to max(nodr_list)).
    max_per_leaf: Maximum spheres per leaf cell.
    max_level:    Maximum tree depth.
    """

    def __init__(
        self,
        positions: np.ndarray,
        nodr_list: list[int],
        k_medium: float,
        p_fmm: Optional[int] = None,
        max_per_leaf: int = 4,
        max_level: int = 8,
        fmm_mode: str = "reference",
        fmm_tol: Optional[float] = None,
        theta: float = 0.55,
        use_precomputed_passes: bool = True,
        d_cache_budget_mb: float = 500.0,
        verbose: bool = True,
    ):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.nsph = len(self.positions)
        self.nodr_list = list(nodr_list)
        self.k_medium = float(k_medium)
        self.fmm_mode = fmm_mode
        self.theta = float(theta)
        self.use_precomputed_passes = use_precomputed_passes
        self.d_cache_budget_mb = d_cache_budget_mb
        self.verbose = verbose

        if fmm_tol is not None:
            self.fmm_tol = float(fmm_tol)
        else:
            self.fmm_tol = 1e-12 if fmm_mode == "reference" else 1e-6

        self.p_fmm = p_fmm if p_fmm is not None else max(nodr_list)
        self.nblk_half = self.p_fmm * (self.p_fmm + 2)
        self.coeff_size = 2 * self.nblk_half

        # Provider cache
        self._cache = _ProviderCache()

        # Dimensionless kR coordinates for translations
        # Convention: positions are ALREADY in kR units (dimensionless),
        # exactly as MultisphereRCRSystem stores and uses them.
        # k_medium is kept for reference but NOT multiplied here.
        self.kpos = self.positions.copy()

        # Build Octree on positions directly (already dimensionless kR coordinates)
        self.tree = Octree(
            self.positions,
            max_per_leaf=max_per_leaf,
            max_level=max_level,
            theta=self.theta,
        )

        # Precompute adjacent leaf mappings (O(N log N) total time)
        self.leaf_adjacent_leaves = {}
        for leaf in self.tree.leaves:
            self.leaf_adjacent_leaves[id(leaf)] = self._find_near_field_leaves(leaf)

        # These will be computed on-demand in pair_stats() for backwards compatibility,
        # avoiding O(N^2) memory and initialization overhead.
        self._near_pairs = None
        self._far_pairs = None

        # Precompute cell truncation orders for each level of the tree using the high-precision formula.
        # Levels 0 and 1 are never used for translations, so we bound their order
        # to self.p_fmm to prevent calculating and pre-warming extremely large, unused orders.
        max_level_count = len(self.tree.levels)
        self.p_levels = []
        log10_eps = -np.log10(self.fmm_tol)
        for l in range(max_level_count):
            if l < 2:
                self.p_levels.append(self.p_fmm)
            else:
                half_l = self.tree.root.half / (2.0 ** l)
                X_l = np.sqrt(3.0) * half_l
                P_l = max(self.p_fmm, int(np.ceil(X_l + 1.6 * (log10_eps ** (2.0 / 3.0)) * (X_l ** (1.0 / 3.0)) + log10_eps)))
                if self.fmm_mode == "speed":
                    P_l = min(P_l, 20)
                self.p_levels.append(P_l)

        # Pre-initialize provider cache for only the specific orders actually used to be thread-safe.
        # This only includes levels >= 2 and the base p_fmm order.
        distinct_orders = set(self.p_levels[2:] + [self.p_fmm]) if len(self.p_levels) > 2 else {self.p_fmm}
        for p in distinct_orders:
            self._cache.rot(p)
            self._cache.trans(p)

        # Cache tran_coef_arr by order
        self.tran_coef_arr_by_order = {}
        for p in distinct_orders:
            self.tran_coef_arr_by_order[p] = self._cache.trans(p).tran_coef_arr

        # Flatten octree for JIT kernels
        flat = self.tree.flatten_tree(self.p_levels, self.p_fmm, self.leaf_adjacent_leaves)
        self.cell_centers = flat["cell_centers"]
        self.cell_half = flat["cell_half"]
        self.cell_level = flat["cell_level"]
        self.cell_parent = flat["cell_parent"]
        self.cell_children = flat["cell_children"]
        self.leaf_spheres_offset = flat["leaf_spheres_offset"]
        self.leaf_spheres_count = flat["leaf_spheres_count"]
        self.leaf_spheres_indices = flat["leaf_spheres_indices"]
        self.cell_offsets = flat["cell_offsets"]
        self.total_coeff_size = flat["total_coeff_size"]
        self.m2l_pairs_level = flat["m2l_pairs_level"]
        self.near_pairs = flat["near_pairs"]
        self.cell_to_idx = flat["cell_to_idx"]

        # Preallocated buffers for apply_batch (reused across calls)
        self._X_padded_cache = None
        self._multipole_coeffs_cache = None
        self._local_coeffs_cache = None
        self._Y_far_padded_cache = None

        if self.use_precomputed_passes:
            self._build_precomputed_cache()

    def _find_near_field_leaves(self, target: Cell) -> list[Cell]:
        """Find all leaf cells in the near-field of target by traversing the tree
        with pruning to skip cells covered by M2L. This is performed once during initialization.
        """
        near_leaves = []
        
        # Precompute target's ancestor map: level -> Cell
        ancestors = {}
        curr = target
        while curr is not None:
            ancestors[curr.level] = curr
            curr = curr.parent

        def traverse(cell: Cell):
            # If cell is at level >= 2, check target's ancestor's interaction list
            if cell.level >= 2:
                target_anc = ancestors.get(cell.level)
                if target_anc is not None:
                    # If cell is in target_anc's interaction list, it's covered by M2L
                    # We can prune the traversal here.
                    if cell in target_anc.interaction_list:
                        return
            
            if not cell.children:
                # It's a leaf!
                if cell is not target:
                    near_leaves.append(cell)
            else:
                for child in cell.children:
                    traverse(child)

        traverse(self.tree.root)
        return near_leaves

    def _build_precomputed_cache(self) -> None:
        """Eagerly precomputes all translation edges and caching structures for the 6 passes:
        P2M, M2M, M2L, L2L, L2P, Near-field.
        """
        # 1. Initialize empty cache lists for the 6 passes
        self.p2m_cache: list[PassGroupCache] = []
        self.m2m_cache: list[PassGroupCache] = []
        self.m2l_cache: list[PassGroupCache] = []
        self.l2l_cache: list[PassGroupCache] = []
        self.l2p_cache: list[PassGroupCache] = []
        self.near_cache: list[PassGroupCache] = []

        # We will collect edges for each pass, grouped by (level, work_order).
        p2m_groups_raw = {}
        m2m_groups_raw = {}
        m2l_groups_raw = {}
        l2l_groups_raw = {}
        l2p_groups_raw = {}
        near_groups_raw = {}

        # --- P2M Edges ---
        for cell in self.tree.leaves:
            cell_idx = self.cell_to_idx[id(cell)]
            cell_offset = self.cell_offsets[cell_idx]
            P_l = self.p_levels[cell.level]
            for idx in cell.sphere_indices:
                tvec = cell.center - self.kpos[idx]
                key = (cell.level, P_l)
                if key not in p2m_groups_raw:
                    p2m_groups_raw[key] = []
                p2m_groups_raw[key].append({
                    "src_idx": idx,
                    "tgt_idx": cell_idx,
                    "src_offset": 0,
                    "tgt_offset": cell_offset,
                    "src_order": self.p_fmm,
                    "tgt_order": P_l,
                    "work_order": P_l,
                    "tvec": tvec,
                })

        # --- M2M Edges ---
        for level in range(len(self.tree.levels) - 1, 1, -1):
            if level > 2:
                parent_level = level - 1
                P_parent = self.p_levels[parent_level]
                for parent_cell in self.tree.levels[parent_level]:
                    parent_idx = self.cell_to_idx[id(parent_cell)]
                    parent_offset = self.cell_offsets[parent_idx]
                    for child in parent_cell.children:
                        child_idx = self.cell_to_idx[id(child)]
                        child_offset = self.cell_offsets[child_idx]
                        tvec = parent_cell.center - child.center
                        
                        # Group key: (child.level, work_order)
                        key = (child.level, P_parent)
                        if key not in m2m_groups_raw:
                            m2m_groups_raw[key] = []
                        m2m_groups_raw[key].append({
                            "src_idx": child_idx,
                            "tgt_idx": parent_idx,
                            "src_offset": child_offset,
                            "tgt_offset": parent_offset,
                            "src_order": self.p_levels[child.level],
                            "tgt_order": P_parent,
                            "work_order": P_parent,
                            "tvec": tvec,
                        })

        # --- M2L Edges ---
        for level in range(2, len(self.tree.levels)):
            P_l = self.p_levels[level]
            m2l_pairs = self.m2l_pairs_level[level]
            if m2l_pairs.shape[0] > 0:
                for idx_pair in range(m2l_pairs.shape[0]):
                    src_idx = m2l_pairs[idx_pair, 0]
                    tgt_idx = m2l_pairs[idx_pair, 1]
                    tvec = self.cell_centers[tgt_idx] - self.cell_centers[src_idx]
                    
                    key = (level, P_l)
                    if key not in m2l_groups_raw:
                        m2l_groups_raw[key] = []
                    m2l_groups_raw[key].append({
                        "src_idx": src_idx,
                        "tgt_idx": tgt_idx,
                        "src_offset": self.cell_offsets[src_idx],
                        "tgt_offset": self.cell_offsets[tgt_idx],
                        "src_order": P_l,
                        "tgt_order": P_l,
                        "work_order": P_l,
                        "tvec": tvec,
                    })

        # --- L2L Edges ---
        for level in range(2, len(self.tree.levels)):
            P_l = self.p_levels[level]
            child_level = level + 1
            if child_level < len(self.tree.levels):
                P_child = self.p_levels[child_level]
                for parent_cell in self.tree.levels[level]:
                    parent_idx = self.cell_to_idx[id(parent_cell)]
                    parent_offset = self.cell_offsets[parent_idx]
                    for child in parent_cell.children:
                        child_idx = self.cell_to_idx[id(child)]
                        child_offset = self.cell_offsets[child_idx]
                        tvec = child.center - parent_cell.center
                        
                        # Group key: (parent.level, work_order)
                        key = (parent_cell.level, P_l)
                        if key not in l2l_groups_raw:
                            l2l_groups_raw[key] = []
                        l2l_groups_raw[key].append({
                            "src_idx": parent_idx,
                            "tgt_idx": child_idx,
                            "src_offset": parent_offset,
                            "tgt_offset": child_offset,
                            "src_order": P_l,
                            "tgt_order": P_child,
                            "work_order": P_l,
                            "tvec": tvec,
                        })

        # --- L2P Edges ---
        for cell in self.tree.leaves:
            cell_idx = self.cell_to_idx[id(cell)]
            cell_offset = self.cell_offsets[cell_idx]
            P_l = self.p_levels[cell.level]
            for idx in cell.sphere_indices:
                tvec = self.kpos[idx] - cell.center
                key = (cell.level, P_l)
                if key not in l2p_groups_raw:
                    l2p_groups_raw[key] = []
                l2p_groups_raw[key].append({
                    "src_idx": cell_idx,
                    "tgt_idx": idx,
                    "src_offset": cell_offset,
                    "tgt_offset": 0,
                    "src_order": P_l,
                    "tgt_order": self.p_fmm,
                    "work_order": P_l,
                    "tvec": tvec,
                })

        # --- Near-field Edges ---
        if self.near_pairs.shape[0] > 0:
            for idx_pair in range(self.near_pairs.shape[0]):
                src_idx = self.near_pairs[idx_pair, 0]
                tgt_idx = self.near_pairs[idx_pair, 1]
                tvec = self.kpos[tgt_idx] - self.kpos[src_idx]
                
                # Near-field always runs at p_fmm order
                key = (0, self.p_fmm)
                if key not in near_groups_raw:
                    near_groups_raw[key] = []
                near_groups_raw[key].append({
                    "src_idx": src_idx,
                    "tgt_idx": tgt_idx,
                    "src_offset": 0,
                    "tgt_offset": 0,
                    "src_order": self.p_fmm,
                    "tgt_order": self.p_fmm,
                    "work_order": self.p_fmm,
                    "tvec": tvec,
                })

        def process_groups(groups_raw: dict, radial_kind: int = 0) -> list[PassGroupCache]:
            caches = []
            for (level, work_order), edges in groups_raw.items():
                if not edges:
                    continue
                
                # Assert homogeneity across all edges in the group
                src_order = edges[0]["src_order"]
                tgt_order = edges[0]["tgt_order"]
                for edge in edges:
                    assert edge["src_order"] == src_order, "Inconsistent src_order in group"
                    assert edge["tgt_order"] == tgt_order, "Inconsistent tgt_order in group"
                    assert edge["work_order"] == work_order, "Inconsistent work_order in group"

                # Precompute Legendre polynomials dc (size 2 * work_order + 1)
                fnr = _precompute_fnr(2 * work_order)
                dc = _mstm_normalizedlegendre_zero_jit(1.0, 2 * work_order, fnr)

                # Get tran_coef_arr
                tran_coef_arr = self._cache.trans(work_order).tran_coef_arr

                # Instantiate PassGroupCache
                g_cache = PassGroupCache(level, work_order, dc, tran_coef_arr)
                g_cache.radial_kind = radial_kind
                num_edges = len(edges)

                # Build edges_int and edges_geom
                g_cache.edges_int = np.empty((num_edges, 8), dtype=np.int32)
                g_cache.edges_geom = np.empty((num_edges, 3), dtype=np.float64)

                # Offsets arrays
                g_cache.phase_offsets = np.arange(0, num_edges + 1, dtype=np.int64) * (2 * work_order + 1)
                g_cache.radial_offsets = np.arange(0, num_edges + 1, dtype=np.int64) * (2 * work_order + 1)

                # Allocate phase and radial flat arrays
                g_cache.phase_pos_flat = np.empty(num_edges * (2 * work_order + 1), dtype=np.complex128)
                g_cache.phase_neg_flat = np.empty(num_edges * (2 * work_order + 1), dtype=np.complex128)
                g_cache.radial_flat = np.empty(num_edges * (2 * work_order + 1), dtype=np.complex128)

                m_vals = np.arange(-work_order, work_order + 1)

                for e, edge in enumerate(edges):
                    tvec = edge["tvec"]
                    r = float(np.linalg.norm(tvec))
                    
                    if r < 1e-14:
                        theta = 0.0
                        phi = 0.0
                        flags = 1
                    else:
                        theta = float(np.arccos(np.clip(tvec[2] / r, -1.0, 1.0)))
                        phi = float(np.arctan2(tvec[1], tvec[0]))
                        flags = 0

                    g_cache.edges_int[e] = [
                        edge["src_idx"],
                        edge["tgt_idx"],
                        edge["src_offset"],
                        edge["tgt_offset"],
                        edge["src_order"],
                        edge["tgt_order"],
                        edge["work_order"],
                        flags
                    ]
                    g_cache.edges_geom[e] = [r, theta, phi]

                    # Populate phase arrays
                    p_start = e * (2 * work_order + 1)
                    g_cache.phase_pos_flat[p_start : p_start + 2 * work_order + 1] = np.exp(1j * m_vals * phi)
                    g_cache.phase_neg_flat[p_start : p_start + 2 * work_order + 1] = np.exp(-1j * m_vals * phi)

                    # Populate radial array using spherical_jn (Bessel) or evaluate_recurrence_hankel_scalar_jit (Hankel)
                    r_start = e * (2 * work_order + 1)
                    if radial_kind == 0:
                        radial_vals = np.array([sp.spherical_jn(w, r) for w in range(2 * work_order + 1)], dtype=np.complex128)
                    else:
                        from fmm.fmm_kernels import evaluate_recurrence_hankel_scalar_jit
                        radial_vals = evaluate_recurrence_hankel_scalar_jit(2 * work_order, r)
                    g_cache.radial_flat[r_start : r_start + 2 * work_order + 1] = radial_vals

                caches.append(g_cache)
            return caches

        # Process the 6 passes
        self.p2m_cache = process_groups(p2m_groups_raw, radial_kind=0)
        self.m2m_cache = process_groups(m2m_groups_raw, radial_kind=0)
        self.m2l_cache = process_groups(m2l_groups_raw, radial_kind=1)
        self.l2l_cache = process_groups(l2l_groups_raw, radial_kind=0)
        self.l2p_cache = process_groups(l2p_groups_raw, radial_kind=0)
        self.near_cache = process_groups(near_groups_raw, radial_kind=1)

        # 2. Memory budget management for Wigner-D blocks
        all_groups = []
        for pass_name, cache in [
            ("p2m", self.p2m_cache),
            ("m2m", self.m2m_cache),
            ("m2l", self.m2l_cache),
            ("l2l", self.l2l_cache),
            ("l2p", self.l2p_cache),
            ("near", self.near_cache)
        ]:
            for group in cache:
                num_edges = group.edges_int.shape[0]
                work_order = group.work_order
                
                block_sum = 0
                for n in range(1, work_order + 1):
                    block_sum += (2 * n + 1) ** 2
                
                priority = num_edges * block_sum
                memory_bytes = num_edges * block_sum * 8
                memory_mb = memory_bytes / (1024.0 * 1024.0)
                
                all_groups.append({
                    "priority": priority,
                    "memory_mb": memory_mb,
                    "group": group,
                    "block_sum": block_sum
                })

        # Sort all groups by priority in descending order
        all_groups.sort(key=lambda x: x["priority"], reverse=True)

        # Apply budget
        total_mb = 0.0
        cached_count = 0
        total_count = len(all_groups)
        
        for item in all_groups:
            group = item["group"]
            memory_mb = item["memory_mb"]
            block_sum = item["block_sum"]
            num_edges = group.edges_int.shape[0]
            
            if total_mb + memory_mb <= self.d_cache_budget_mb:
                group.has_d_cache = True
                total_mb += memory_mb
                cached_count += 1
                
                group.d_offsets = np.arange(0, num_edges + 1, dtype=np.int64) * block_sum
                group.d_blocks_flat = np.empty(num_edges * block_sum, dtype=np.float64)
                for e in range(num_edges):
                    theta = group.edges_geom[e, 1]
                    d_blocks_edge = _build_cached_d_jit(group.work_order, theta, FAC_TABLE)
                    d_start = e * block_sum
                    group.d_blocks_flat[d_start : d_start + block_sum] = d_blocks_edge
            else:
                group.has_d_cache = False
                group.d_offsets = np.arange(0, num_edges + 1, dtype=np.int64) * block_sum
                group.d_blocks_flat = np.empty(0, dtype=np.float64)

        # Diagnostic message
        if self.verbose:
            if cached_count < total_count:
                logger.info(f"[FMM Operator] Cached {cached_count}/{total_count} translation groups in Wigner-D memory budget ({total_mb:.1f} MB / {self.d_cache_budget_mb:.1f} MB budget).")
                logger.info("[FMM Operator] Remaining groups will compute Wigner-D rotation blocks on-the-fly. Increase 'd_cache_budget_mb' to cache all groups.")
            else:
                logger.info(f"[FMM Operator] Successfully cached all {total_count} translation groups in Wigner-D memory budget ({total_mb:.1f} MB).")

    def _build_pair_lists(self) -> None:
        """Classify every ordered sphere pair (i, j) i != j as near or far.

        Near-field: i and j share the same leaf, or their leaf cells are adjacent.
        Far-field:  j's leaf is in the interaction list of i's leaf.
        """
        self._near_pairs = set()
        self._far_pairs = set()
        # Map sphere -> its leaf cell
        sphere_to_leaf: dict[int, Cell] = {}
        for leaf in self.tree.leaves:
            for idx in leaf.sphere_indices:
                sphere_to_leaf[idx] = leaf

        for i in range(self.nsph):
            for j in range(self.nsph):
                if i == j:
                    continue
                leaf_i = sphere_to_leaf[i]
                leaf_j = sphere_to_leaf[j]

                # Same cell or adjacent cell -> near-field
                if leaf_i is leaf_j or leaf_i.is_adjacent(leaf_j):
                    self._near_pairs.add((i, j))
                else:
                    # Check if leaf_j (or any ancestor) is in leaf_i's
                    # interaction list (including all ancestor interaction lists
                    # that contain leaf_j)
                    if self._is_interaction_pair(leaf_i, leaf_j):
                        self._far_pairs.add((i, j))
                    else:
                        # Fallback: if not classified, treat as near-field
                        # to guarantee correctness (no pair is skipped)
                        self._near_pairs.add((i, j))

    def _is_interaction_pair(self, leaf_i: Cell, leaf_j: Cell) -> bool:
        """Check if leaf_j is in the interaction list of leaf_i at any level."""
        # Check interaction list at leaf level
        il_ids = {id(c) for c in leaf_i.interaction_list}
        if id(leaf_j) in il_ids:
            return True
        # Also check ancestor interaction lists that might cover leaf_j
        # by walking up the tree and checking if leaf_j is in sub-trees
        # of cells in the interaction list
        for src_cell in leaf_i.interaction_list:
            if self._cell_contains_leaf(src_cell, leaf_j):
                return True
        return False

    def _cell_contains_leaf(self, cell: Cell, leaf: Cell) -> bool:
        """Check if leaf is a descendant of cell."""
        if cell is leaf:
            return True
        for child in cell.children:
            if self._cell_contains_leaf(child, leaf):
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def _pad_coefficients(self, x: np.ndarray, p_in: int, p_out: int) -> np.ndarray:
        """Pad vector of order p_in to order p_out."""
        if p_in == p_out:
            return x
        nblk_in = p_in * (p_in + 2)
        nblk_out = p_out * (p_out + 2)
        x_out = np.zeros(2 * nblk_out, dtype=np.complex128)
        x_out[:nblk_in] = x[:nblk_in]
        x_out[nblk_out:nblk_out + nblk_in] = x[nblk_in:]
        return x_out

    def _slice_coefficients(self, x: np.ndarray, p_in: int, p_out: int) -> np.ndarray:
        """Truncate vector of order p_in to order p_out."""
        if p_in == p_out:
            return x
        nblk_in = p_in * (p_in + 2)
        nblk_out = p_out * (p_out + 2)
        x_out = np.zeros(2 * nblk_out, dtype=np.complex128)
        x_out[:nblk_out] = x[:nblk_out]
        x_out[nblk_out:] = x[nblk_in:nblk_in + nblk_out]
        return x_out

    def apply(self, X_blocks: list[np.ndarray]) -> list[np.ndarray]:
        """Apply FMM operator to a single column (delegates to apply_batch)."""
        return self.apply_batch(X_blocks)

    def apply_batch(self, X_blocks_batch: list[np.ndarray]) -> list[np.ndarray]:
        """Apply FMM operator: near-field direct + hierarchical far-field traversal for batch of RHS columns.

        Args:
            X_blocks_batch: List of N arrays, each shape (2 * nblk_half_j, B) or (2 * nblk_half_j,).

        Returns:
            Y_blocks_batch: List of N arrays, each shape (2 * nblk_half_i, B) or (2 * nblk_half_i,).
        """
        nsph = self.nsph
        
        # Determine batch size B
        B = X_blocks_batch[0].shape[1] if len(X_blocks_batch) > 0 and X_blocks_batch[0].ndim > 1 else 1

        # 1. Reuse or allocate buffers
        if (self._multipole_coeffs_cache is None or 
            self._multipole_coeffs_cache.shape[1] != B):
            self._X_padded_cache = np.zeros((nsph, self.coeff_size, B), dtype=np.complex128)
            self._multipole_coeffs_cache = np.zeros((self.total_coeff_size, B), dtype=np.complex128)
            self._local_coeffs_cache = np.zeros((self.total_coeff_size, B), dtype=np.complex128)
            self._Y_far_padded_cache = np.zeros((nsph, self.coeff_size, B), dtype=np.complex128)
        else:
            self._X_padded_cache.fill(0.0)
            self._multipole_coeffs_cache.fill(0.0)
            self._local_coeffs_cache.fill(0.0)
            self._Y_far_padded_cache.fill(0.0)

        X_padded = self._X_padded_cache
        multipole_coeffs = self._multipole_coeffs_cache
        local_coeffs = self._local_coeffs_cache
        Y_far_padded = self._Y_far_padded_cache

        # Pad inputs to uniform FMM order
        for j, xj in enumerate(X_blocks_batch):
            nblk_j = self.nodr_list[j] * (self.nodr_list[j] + 2)
            if xj.ndim > 1:
                X_padded[j, :nblk_j, :] = xj[:nblk_j, :]
                X_padded[j, self.nblk_half : self.nblk_half + nblk_j, :] = xj[nblk_j:, :]
            else:
                X_padded[j, :nblk_j, 0] = xj[:nblk_j]
                X_padded[j, self.nblk_half : self.nblk_half + nblk_j, 0] = xj[nblk_j:]

        # 3. Upward Pass: P2M & M2M (in Python, O(N) operations)
        if self.use_precomputed_passes:
            # P2M: Dispatch groups
            for group in self.p2m_cache:
                p2m_pass_group_jit(
                    group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                    group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                    group.phase_neg_flat, multipole_coeffs, X_padded, self.p_fmm, group.dc, group.tran_coef_arr,
                    group.has_d_cache, FAC_TABLE
                )
            
            # M2M: Dispatch groups by level from max_level down to 3 (bottom-up)
            max_level = len(self.tree.levels) - 1
            for level in range(max_level, 2, -1):
                for group in self.m2m_cache:
                    if group.level == level:
                        m2m_pass_group_jit(
                            group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                            group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                            group.phase_neg_flat, multipole_coeffs, group.dc, group.tran_coef_arr,
                            group.has_d_cache, FAC_TABLE
                        )
        else:
            for level in range(len(self.tree.levels) - 1, 1, -1):
                P_l = self.p_levels[level]
                coeff_size_l = 2 * P_l * (P_l + 2)
                
                for cell in self.tree.levels[level]:
                    cell_idx = self.cell_to_idx[id(cell)]
                    cell_offset = self.cell_offsets[cell_idx]
                    
                    if not cell.children:
                        # Leaf cell: P2M (regular=True)
                        for idx in cell.sphere_indices:
                            tvec = cell.center - self.kpos[idx]
                            for b in range(B):
                                x_sphere_padded = self._pad_coefficients(X_padded[idx, :, b], self.p_fmm, P_l)
                                translated = _rcr_translate(
                                    tvec, x_sphere_padded, P_l, self._cache, regular=True
                                )
                                multipole_coeffs[cell_offset : cell_offset + coeff_size_l, b] += translated
                
                # M2M (child to parent cell)
                if level > 2:
                    parent_level = level - 1
                    P_parent = self.p_levels[parent_level]
                    coeff_size_parent = 2 * P_parent * (P_parent + 2)
                    
                    for parent_cell in self.tree.levels[parent_level]:
                        parent_idx = self.cell_to_idx[id(parent_cell)]
                        parent_offset = self.cell_offsets[parent_idx]
                        
                        for child in parent_cell.children:
                            child_idx = self.cell_to_idx[id(child)]
                            child_offset = self.cell_offsets[child_idx]
                            child_size = 2 * self.p_levels[level] * (self.p_levels[level] + 2)
                            
                            tvec = parent_cell.center - child.center
                            
                            for b in range(B):
                                child_mp = multipole_coeffs[child_offset : child_offset + child_size, b]
                                child_mp_padded = self._pad_coefficients(child_mp, self.p_levels[level], P_parent)
                                translated = _rcr_translate(
                                    tvec, child_mp_padded, P_parent, self._cache, regular=True
                                )
                                multipole_coeffs[parent_offset : parent_offset + coeff_size_parent, b] += translated

        # 4. Horizontal Pass: M2L (precomputed or JIT fallback)
        max_order = max(self.p_levels[2:] + [self.p_fmm])
        fnr = self._cache.trans(max_order).fnr

        if self.use_precomputed_passes:
            for level in range(2, len(self.tree.levels)):
                for group in self.m2l_cache:
                    if group.level == level:
                        m2l_pass_group_jit(
                            group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                            group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                            group.phase_neg_flat, multipole_coeffs, local_coeffs, group.dc, group.tran_coef_arr,
                            group.has_d_cache, FAC_TABLE
                        )
        else:
            for level in range(2, len(self.tree.levels)):
                P_l = self.p_levels[level]
                m2l_pairs = self.m2l_pairs_level[level]
                if m2l_pairs.shape[0] > 0:
                    tran_coef_arr = self.tran_coef_arr_by_order[P_l]
                    horizontal_pass_level_batch_jit(
                        m2l_pairs,
                        multipole_coeffs,
                        local_coeffs,
                        self.cell_offsets,
                        self.cell_centers,
                        P_l,
                        tran_coef_arr,
                        FAC_TABLE,
                        fnr,
                    )

        # 5. Downward Pass: L2L & L2P (in Python or JIT group pass)
        if self.use_precomputed_passes:
            max_level = len(self.tree.levels) - 1
            # L2L: Dispatch groups by level from 2 up to max_level - 1 (top-down)
            for level in range(2, max_level):
                for group in self.l2l_cache:
                    if group.level == level:
                        l2l_pass_group_jit(
                            group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                            group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                            group.phase_neg_flat, local_coeffs, group.dc, group.tran_coef_arr,
                            group.has_d_cache, FAC_TABLE
                        )
            
            # L2P: Dispatch groups
            for group in self.l2p_cache:
                l2p_pass_group_jit(
                    group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                    group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                    group.phase_neg_flat, local_coeffs, Y_far_padded, group.dc, group.tran_coef_arr,
                    group.has_d_cache, FAC_TABLE
                )
        else:
            for level in range(2, len(self.tree.levels)):
                P_l = self.p_levels[level]
                child_level = level + 1
                if child_level < len(self.tree.levels):
                    P_child = self.p_levels[child_level]
                    coeff_size_child = 2 * P_child * (P_child + 2)
                    
                    for parent_cell in self.tree.levels[level]:
                        parent_idx = self.cell_to_idx[id(parent_cell)]
                        parent_offset = self.cell_offsets[parent_idx]
                        
                        for child in parent_cell.children:
                            child_idx = self.cell_to_idx[id(child)]
                            child_offset = self.cell_offsets[child_idx]
                            
                            tvec = child.center - parent_cell.center
                            
                            for b in range(B):
                                parent_local = local_coeffs[parent_offset : parent_offset + 2 * P_l * (P_l + 2), b]
                                translated = _rcr_translate(
                                    tvec, parent_local, P_l, self._cache, regular=True
                                )
                                local_child = self._slice_coefficients(translated, P_l, P_child)
                                local_coeffs[child_offset : child_offset + coeff_size_child, b] += local_child

            # Leaf to Spheres: L2P
            for cell in self.tree.leaves:
                cell_idx = self.cell_to_idx[id(cell)]
                cell_offset = self.cell_offsets[cell_idx]
                P_l = self.p_levels[cell.level]
                coeff_size_l = 2 * P_l * (P_l + 2)
                
                for idx in cell.sphere_indices:
                    tvec = self.kpos[idx] - cell.center
                    for b in range(B):
                        local_cell = local_coeffs[cell_offset : cell_offset + coeff_size_l, b]
                        translated = _rcr_translate(
                            tvec, local_cell, P_l, self._cache, regular=True
                        )
                        sliced = self._slice_coefficients(translated, P_l, self.p_fmm)
                        Y_far_padded[idx, :, b] += sliced

        # 6. Near-field Direct pass (precomputed JIT or fallback)
        Y_near_padded = np.zeros((nsph, self.coeff_size, B), dtype=np.complex128)
        if self.use_precomputed_passes and len(self.near_cache) > 0:
            for group in self.near_cache:
                near_field_pass_group_jit(
                    group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                    group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                    group.phase_neg_flat, X_padded, Y_near_padded, group.dc, group.tran_coef_arr,
                    group.has_d_cache, FAC_TABLE
                )
        else:
            tran_coef_arr_fmm = self.tran_coef_arr_by_order[self.p_fmm]
            Y_near_padded = near_field_direct_pass_batch_jit(
                self.near_pairs,
                X_padded,
                self.kpos,
                self.p_fmm,
                self.nblk_half,
                FAC_TABLE,
                tran_coef_arr_fmm,
                fnr,
            )

        # 7. Combine results
        Y_combined_padded = Y_near_padded + Y_far_padded

        # Slice back to per-sphere sizes
        Y_blocks_batch = []
        for i in range(nsph):
            nblk_i = self.nodr_list[i] * (self.nodr_list[i] + 2)
            y_i = np.empty((2 * nblk_i, B), dtype=np.complex128)
            y_i[:nblk_i, :] = Y_combined_padded[i, :nblk_i, :]
            y_i[nblk_i:, :] = Y_combined_padded[i, self.nblk_half : self.nblk_half + nblk_i, :]
            
            if X_blocks_batch[0].ndim == 1:
                Y_blocks_batch.append(y_i[:, 0])
            else:
                Y_blocks_batch.append(y_i)
                
        return Y_blocks_batch

    def apply_grouped(self, X_blocks: list[np.ndarray]) -> list[np.ndarray]:
        """Apply FMM operator with grouping."""
        return self.apply(X_blocks)

    def prewarm_cache(self, max_p: int) -> None:
        """Pre-initialize rotation and translation providers up to max_p."""
        for p in range(1, max_p + 1):
            self._cache.rot(p)
            self._cache.trans(p)

    def translate_rcr_general(
        self,
        tvec: np.ndarray,
        x: np.ndarray,
        source_order: int,
        target_order: int,
        regular: bool = False,
    ) -> np.ndarray:
        """Translate coefficient vector x over arbitrary displacement tvec using RCR."""
        p_max = max(source_order, target_order)
        x_padded = self._pad_coefficients(x, source_order, p_max)
        y_translated = _rcr_translate(tvec, x_padded, p_max, self._cache, regular=regular)
        return self._slice_coefficients(y_translated, p_max, target_order)

    # ------------------------------------------------------------------ #
    #  Padding helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pad_to_fmm(self, X_blocks: list[np.ndarray]) -> list[np.ndarray]:
        """Pad per-sphere coefficient vectors to p_fmm truncation order."""
        result = []
        for j, xj in enumerate(X_blocks):
            nblk_j = self.nodr_list[j] * (self.nodr_list[j] + 2)
            xp = np.zeros(self.coeff_size, dtype=np.complex128)
            xp[:nblk_j] = xj[:nblk_j]
            xp[self.nblk_half:self.nblk_half + nblk_j] = xj[nblk_j:]
            result.append(xp)
        return result

    def _slice_from_fmm(self, Y_padded: list[np.ndarray]) -> list[np.ndarray]:
        """Slice FMM-order vectors back to per-sphere sizes."""
        result = []
        for i, yi in enumerate(Y_padded):
            nblk_i = self.nodr_list[i] * (self.nodr_list[i] + 2)
            yi_sliced = np.zeros(2 * nblk_i, dtype=np.complex128)
            yi_sliced[:nblk_i] = yi[:nblk_i]
            yi_sliced[nblk_i:] = yi[self.nblk_half:self.nblk_half + nblk_i]
            result.append(yi_sliced)
        return result

    # ------------------------------------------------------------------ #
    #  Statistics                                                          #
    # ------------------------------------------------------------------ #

    def pair_stats(self) -> dict:
        """Return statistics about pair classification."""
        if self._near_pairs is None or self._far_pairs is None:
            self._build_pair_lists()
        n_total = self.nsph * (self.nsph - 1)
        n_near = len(self._near_pairs)
        n_far = len(self._far_pairs)
        n_classified = n_near + n_far
        return {
            "N_spheres": self.nsph,
            "N_pairs_total": n_total,
            "N_near_pairs": n_near,
            "N_far_pairs": n_far,
            "N_classified": n_classified,
            "fraction_far": n_far / n_total if n_total > 0 else 0.0,
            "all_classified": n_classified == n_total,
        }
