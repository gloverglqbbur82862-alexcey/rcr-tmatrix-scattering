#!/usr/bin/env python3
"""FMM Octree: Hierarchical spatial partitioning for O(N log N) multi-sphere solver.

This module builds a minimal Octree over a set of sphere positions, assigns
interaction lists (well-separated, interaction, near-field), and provides
the cell data structures needed for the FMM P2M -> M2M -> M2L -> L2L -> L2P
multi-pass computation.

Design principles:
- Pure Python/NumPy; no compiled dependencies.
- Works for N >= 1; degenerates gracefully at small N.
- Separation criterion: cells are "well-separated" if they are not direct
  neighbors at the same level (standard FMM interaction list definition).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cell:
    """A single cell in the FMM octree.

    Attributes
    ----------
    level:    Depth in the tree (root = 0).
    index:    Linear index of this cell within its level (Morton ordering).
    center:   3D center position of this cell's bounding box.
    half:     Half-width of the bounding box.
    parent:   Parent cell, or None at root.
    children: List of up to 8 child cells.
    sphere_indices: Global sphere indices whose centers lie in this cell.
    interaction_list: Cells in this cell's interaction list
                      (same-level, well-separated from cell's parent's neighbors).
    near_field:       Same-level cells whose bounding boxes touch this cell's
                      bounding box (direct-summation region).
    multipole_coeff:  Complex VSWF multipole coefficients, shape (2*nblk_half,).
                      None before P2M/M2M pass.
    local_coeff:      Complex VSWF local-expansion coefficients, shape (2*nblk_half,).
                      None before M2L/L2L pass.
    """
    level: int
    index: int
    center: np.ndarray
    half: float
    parent: Optional["Cell"] = field(default=None, repr=False)
    children: list["Cell"] = field(default_factory=list, repr=False)
    sphere_indices: list[int] = field(default_factory=list)
    interaction_list: list["Cell"] = field(default_factory=list, repr=False)
    near_field: list["Cell"] = field(default_factory=list, repr=False)
    multipole_coeff: Optional[np.ndarray] = field(default=None, repr=False)
    local_coeff: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    #  Geometric helpers                                                   #
    # ------------------------------------------------------------------ #
    def contains(self, pos: np.ndarray) -> bool:
        """Return True if pos lies inside this cell using a HALF-OPEN interval.

        Convention: [center - half, center + half) in each dimension.
        This guarantees each position belongs to EXACTLY ONE child cell.
        """
        d = pos - self.center
        return bool(np.all((d >= -self.half) & (d < self.half)))

    def is_adjacent(self, other: "Cell") -> bool:
        """Return True if this cell and other are adjacent (touching or overlapping)."""
        d = np.abs(self.center - other.center)
        # Adjacent when distance in each coordinate <= sum of half-widths + epsilon
        return bool(np.all(d <= self.half + other.half + 1e-10 * self.half))

    def displacement_to(self, other: "Cell") -> np.ndarray:
        """Displacement vector from this cell center to other cell center."""
        return other.center - self.center


# ================================================================== #
#  Octree builder                                                      #
# ================================================================== #

class Octree:
    """Build and manage a hierarchical FMM octree over N sphere positions.

    Parameters
    ----------
    positions:    (N, 3) sphere center positions.
    max_per_leaf: Maximum number of spheres in a leaf cell (default 4).
    max_level:    Maximum allowed depth (default 8).

    Attributes
    ----------
    root:        Root cell of the tree.
    levels:      List of lists; levels[d] = all cells at depth d.
    leaves:      All leaf cells (cells with sphere_indices and no children).
    """

    def __init__(self, positions: np.ndarray, max_per_leaf: int = 4, max_level: int = 8, theta: float = 0.55):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.nsph = len(self.positions)
        self.max_per_leaf = max_per_leaf
        self.max_level = max_level
        self.theta = float(theta)

        # Build bounding box
        lo = self.positions.min(axis=0)
        hi = self.positions.max(axis=0)
        # Add small buffer to prevent boundary issues
        margin = np.max(hi - lo) * 0.01 + 1e-10
        center = 0.5 * (lo + hi)
        half = 0.5 * np.max(hi - lo) + margin

        self.root = Cell(
            level=0, index=0, center=center, half=half,
            sphere_indices=list(range(self.nsph))
        )

        self.levels: list[list[Cell]] = [[self.root]]
        self.leaves: list[Cell] = []

        # Recursively build tree
        self._subdivide(self.root)

        # Build interaction lists at every level >= 2
        self._build_interaction_lists()

    # ------------------------------------------------------------------ #
    #  Tree construction                                                   #
    # ------------------------------------------------------------------ #

    def _subdivide(self, cell: Cell) -> None:
        """Recursively subdivide a cell until max_per_leaf or max_level."""
        if (len(cell.sphere_indices) <= self.max_per_leaf or
                cell.level >= self.max_level):
            self.leaves.append(cell)
            return

        half_child = cell.half / 2.0
        child_idx = 0
        offsets = [
            np.array([dx, dy, dz]) * half_child
            for dx in (-1, 1)
            for dy in (-1, 1)
            for dz in (-1, 1)
        ]

        # Ensure levels list is large enough
        child_level = cell.level + 1
        while len(self.levels) <= child_level:
            self.levels.append([])

        for offset in offsets:
            child_center = cell.center + offset
            child = Cell(
                level=child_level,
                index=len(self.levels[child_level]),
                center=child_center,
                half=half_child,
                parent=cell,
            )
            # Assign spheres
            for idx in cell.sphere_indices:
                if child.contains(self.positions[idx]):
                    child.sphere_indices.append(idx)

            if child.sphere_indices:
                cell.children.append(child)
                self.levels[child_level].append(child)
                self._subdivide(child)

    # ------------------------------------------------------------------ #
    #  Interaction list construction                                       #
    # ------------------------------------------------------------------ #

    def _build_interaction_lists(self) -> None:
        """Assign interaction_list and near_field for all cells at each level.

        Standard FMM interaction list definition (with geometric admissibility check):
        - A cell C's near field at level L = same-level cells adjacent to C.
        - A cell C's interaction list at level L = children of C's parent's
          near-field cells that are NOT adjacent to C AND satisfy the geometric
          admissibility criterion.
        """
        for level_idx, level_cells in enumerate(self.levels):
            if level_idx < 2:
                # Root and level-1 cells: everything is "near field"
                for cell in level_cells:
                    cell.near_field = [c for c in level_cells if c is not cell]
                continue

            for cell in level_cells:
                # Near field: same-level cells adjacent to cell
                cell.near_field = [
                    c for c in level_cells
                    if c is not cell and cell.is_adjacent(c)
                ]

                # Interaction list: children of parent's near-field cells
                # that are NOT in cell's own near field AND satisfy geometric admissibility
                if cell.parent is not None:
                    near_set = set(id(c) for c in cell.near_field)
                    near_set.add(id(cell))
                    for parent_neighbor in cell.parent.near_field:
                        for cousin in parent_neighbor.children:
                            if id(cousin) not in near_set:
                                # Geometric admissibility check:
                                d = np.linalg.norm(cell.center - cousin.center)
                                r_sum = 2.0 * np.sqrt(3.0) * cell.half
                                if d > 1e-14 and (r_sum / d) < self.theta:
                                    cell.interaction_list.append(cousin)

    # ------------------------------------------------------------------ #
    #  Traversal utilities                                                 #
    # ------------------------------------------------------------------ #

    def bottom_up_cells(self) -> list[Cell]:
        """Return all cells in bottom-up order (leaves first, root last)."""
        result = []
        for level_cells in reversed(self.levels):
            result.extend(level_cells)
        return result

    def top_down_cells(self) -> list[Cell]:
        """Return all cells in top-down order (root first, leaves last)."""
        result = []
        for level_cells in self.levels:
            result.extend(level_cells)
        return result

    def all_cells(self) -> list[Cell]:
        """Return a flat list of all cells in top-down order."""
        return self.top_down_cells()

    def reset_coefficients(self, coeff_size: int) -> None:
        """Zero-initialise multipole and local coefficients for all cells."""
        for cell in self.all_cells():
            cell.multipole_coeff = np.zeros(coeff_size, dtype=np.complex128)
            cell.local_coeff = np.zeros(coeff_size, dtype=np.complex128)

    def flatten_tree(self, p_levels: list[int], p_fmm: int, leaf_adjacent_leaves: dict) -> dict:
        """Export tree structures as contiguous NumPy arrays for Numba JIT.

        Parameters
        ----------
        p_levels: Cell truncation orders for each level of the tree.
        p_fmm: Base VSWF order.
        leaf_adjacent_leaves: Precomputed map of leaf cell object ID -> list of adjacent leaf cells.

        Returns
        -------
        Dictionary containing all flat arrays.
        """
        all_c = self.all_cells()
        cell_to_idx = {id(c): idx for idx, c in enumerate(all_c)}
        ncells = len(all_c)

        cell_centers = np.empty((ncells, 3), dtype=np.float64)
        cell_half = np.empty(ncells, dtype=np.float64)
        cell_level = np.empty(ncells, dtype=np.int32)
        cell_parent = np.empty(ncells, dtype=np.int32)
        cell_children = np.empty((ncells, 8), dtype=np.int32)
        leaf_spheres_offset = np.empty(ncells, dtype=np.int32)
        leaf_spheres_count = np.empty(ncells, dtype=np.int32)
        cell_offsets = np.empty(ncells, dtype=np.int32)

        curr_offset = 0
        for idx, c in enumerate(all_c):
            cell_offsets[idx] = curr_offset
            coeff_size_c = 2 * p_levels[c.level] * (p_levels[c.level] + 2)
            curr_offset += coeff_size_c

        leaf_spheres_indices = []
        for idx, c in enumerate(all_c):
            if not c.children: # It's a leaf!
                leaf_spheres_offset[idx] = len(leaf_spheres_indices)
                leaf_spheres_count[idx] = len(c.sphere_indices)
                leaf_spheres_indices.extend(c.sphere_indices)
            else:
                leaf_spheres_offset[idx] = -1
                leaf_spheres_count[idx] = 0
        leaf_spheres_indices = np.array(leaf_spheres_indices, dtype=np.int32)

        for idx, c in enumerate(all_c):
            cell_centers[idx] = c.center
            cell_half[idx] = c.half
            cell_level[idx] = c.level
            cell_parent[idx] = cell_to_idx[id(c.parent)] if c.parent is not None else -1
            
            for child_idx in range(8):
                if child_idx < len(c.children):
                    cell_children[idx, child_idx] = cell_to_idx[id(c.children[child_idx])]
                else:
                    cell_children[idx, child_idx] = -1

        m2l_pairs_level = {}
        for l in range(2, len(self.levels)):
            pairs = []
            for tgt_cell in self.levels[l]:
                tgt_idx = cell_to_idx[id(tgt_cell)]
                for src_cell in tgt_cell.interaction_list:
                    src_idx = cell_to_idx[id(src_cell)]
                    pairs.append((src_idx, tgt_idx))
            if pairs:
                m2l_pairs_level[l] = np.array(pairs, dtype=np.int32)
            else:
                m2l_pairs_level[l] = np.empty((0, 2), dtype=np.int32)

        near_pairs = []
        for cell in self.leaves:
            for i in cell.sphere_indices:
                for j in cell.sphere_indices:
                    if i != j:
                        near_pairs.append((j, i))
                adj_leaves = leaf_adjacent_leaves.get(id(cell), [])
                for adj_leaf in adj_leaves:
                    for j in adj_leaf.sphere_indices:
                        near_pairs.append((j, i))
        max_level_val = len(self.levels) - 1
        level_cells_offset = np.zeros(max_level_val + 1, dtype=np.int32)
        level_cells_count = np.zeros(max_level_val + 1, dtype=np.int32)
        curr = 0
        for l in range(max_level_val + 1):
            level_cells_offset[l] = curr
            level_cells_count[l] = len(self.levels[l])
            curr += len(self.levels[l])

        near_pairs = np.array(near_pairs, dtype=np.int32) if near_pairs else np.empty((0, 2), dtype=np.int32)

        return {
            "cell_centers": cell_centers,
            "cell_half": cell_half,
            "cell_level": cell_level,
            "cell_parent": cell_parent,
            "cell_children": cell_children,
            "leaf_spheres_offset": leaf_spheres_offset,
            "leaf_spheres_count": leaf_spheres_count,
            "leaf_spheres_indices": leaf_spheres_indices,
            "cell_offsets": cell_offsets,
            "total_coeff_size": curr_offset,
            "m2l_pairs_level": m2l_pairs_level,
            "near_pairs": near_pairs,
            "cell_to_idx": cell_to_idx,
            "level_cells_offset": level_cells_offset,
            "level_cells_count": level_cells_count,
        }

    # ------------------------------------------------------------------ #
    #  Summary / diagnostics                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a human-readable summary of the tree structure."""
        lines = [
            f"Octree summary:",
            f"  Spheres: {self.nsph}",
            f"  Max per leaf: {self.max_per_leaf}",
            f"  Tree depth: {len(self.levels) - 1}",
            f"  Total cells: {sum(len(lv) for lv in self.levels)}",
            f"  Leaf cells: {len(self.leaves)}",
        ]
        for d, lv in enumerate(self.levels):
            nonempty = sum(1 for c in lv if c.sphere_indices)
            lines.append(f"  Level {d}: {len(lv)} cells, {nonempty} non-empty")
        return "\n".join(lines)
