from __future__ import annotations

import random
from typing import Optional, Tuple, List, Union

import torch
from torch import Tensor
from torch import nn

# Optional PyG import (required for actual execution; we stub classes for typing)
try:
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    PYG_AVAILABLE = False

    # Stubs so editors/type-checkers know these names
    class Data:  # type: ignore
        pass

    class Batch:  # type: ignore
        pass


class GraphCLGraphTransform(nn.Module):
    """
    GraphCL-style graph augmentations that produce TWO independent views.

    Augmentations (You et al., NeurIPS 2020):
        - Node Dropping
        - Edge Perturbation (random add/remove)
        - Attribute Masking
        - Subgraph (node subset via Random Walk)  <-- modified to RW

    Behavior:
        For each input graph sample, we randomly choose one augmentation for view-1
        and (independently) one augmentation for view-2 from the active set, and
        apply them with their configured ratios.

    Notes:
        * Works on single `Data` or on `Batch`; if given a `Batch`, it returns a pair of `Batch`.
        * If an augmentation is deactivated (ratio=0.0), it won't be sampled.
        * All ops are per-graph; for batches we split to a list, augment, then re-batch.

    Args:
        node_drop_ratio: Fraction of nodes to drop.
        edge_perturb_ratio: Fraction of edges to perturb (remove+add of ~same count).
        attr_mask_ratio: Fraction of nodes to mask (features set to 0).
        subgraph_ratio: Fraction of nodes to keep for subgraph (used by RW target size).
        allow_empty: If False, guarantees at least 1 node remains after node/subgraph drop.
        undirected_hint: If True, attempts to keep added edges approximately symmetric.
        rng: Optional torch.Generator for reproducibility.
    """

    def __init__(
        self,
        node_drop_ratio: float = 0.2,
        edge_perturb_ratio: float = 0.2,
        attr_mask_ratio: float = 0.2,
        subgraph_ratio: float = 0.2,
        allow_empty: bool = False,
        undirected_hint: bool = True,
        rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if not PYG_AVAILABLE:  # pragma: no cover
            raise ImportError("torch_geometric is required for GraphCLGraphTransform.")

        self.node_drop_ratio = float(node_drop_ratio)
        self.edge_perturb_ratio = float(edge_perturb_ratio)
        self.attr_mask_ratio = float(attr_mask_ratio)
        self.subgraph_ratio = float(subgraph_ratio)
        self.allow_empty = bool(allow_empty)
        self.undirected_hint = bool(undirected_hint)
        self._rng = rng  # optional torch.Generator

        self._ops: List[str] = []
        if self.node_drop_ratio > 0:
            self._ops.append("node_drop")
        if self.edge_perturb_ratio > 0:
            self._ops.append("edge_perturb")
        if self.attr_mask_ratio > 0:
            self._ops.append("attr_mask")
        if self.subgraph_ratio > 0:
            self._ops.append("subgraph")
        if not self._ops:
            # If all ratios are zero, default to identity
            self._ops.append("identity")

    # ---- Public API ---------------------------------------------------------

    def forward(self, data_or_batch: Union[Data, Batch]) -> Tuple[Union[Data, Batch], Union[Data, Batch]]:
        """
        Create two independently augmented views for the given sample or batch.

        Args:
            data_or_batch: `Data` (single graph) OR `Batch` (multiple graphs).

        Returns:
            (view1, view2): same outer type as the input (Data or Batch).
        """
        if isinstance(data_or_batch, Batch):
            data_list = data_or_batch.to_data_list()
            v1_list = [self._augment_one(d, self._sample_op()) for d in data_list]
            v2_list = [self._augment_one(d, self._sample_op()) for d in data_list]
            return Batch.from_data_list(v1_list), Batch.from_data_list(v2_list)
        else:
            return (
                self._augment_one(data_or_batch, self._sample_op()),
                self._augment_one(data_or_batch, self._sample_op()),
            )

    # ---- Internals ----------------------------------------------------------

    def _sample_op(self) -> str:
        # Randomly choose one augmentation for a view
        if self._rng is not None:
            idx = torch.randint(len(self._ops), (1,), generator=self._rng).item()
        else:
            idx = random.randrange(len(self._ops))
        return self._ops[idx]

    def _augment_one(self, data: Data, op: str) -> Data:
        if op == "node_drop":
            return self._node_drop(data, self.node_drop_ratio)
        if op == "edge_perturb":
            return self._edge_perturb(data, self.edge_perturb_ratio)
        if op == "attr_mask":
            return self._attr_mask(data, self.attr_mask_ratio)
        if op == "subgraph":
            return self._subgraph_rw(data, self.subgraph_ratio)  # <-- RW-based subgraph
        # identity fallback
        return data

    # ---- Primitive ops ------------------------------------------------------

    def _node_drop(self, data: Data, ratio: float) -> Data:
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0) if x is not None else int(data.num_nodes)

        keep_count = max(1 if not self.allow_empty else 0, int(round((1.0 - ratio) * num_nodes)))
        if keep_count >= num_nodes:
            return data
        keep_idx = self._randperm(num_nodes)[:keep_count]
        keep_idx, _ = torch.sort(keep_idx)

        return self._induce_by_nodes(data, keep_idx)

    def _edge_perturb(self, data: Data, ratio: float) -> Data:
        # Randomly remove a fraction of edges and add approximately the same number of random edges.
        if data.edge_index is None or data.edge_index.numel() == 0:
            return data

        edge_index = data.edge_index
        num_nodes = int(data.num_nodes)
        E = edge_index.size(1)
        if E == 0:
            return data

        # Remove
        remove_E = min(E, int(round(ratio * E)))
        if remove_E > 0:
            perm = self._randperm(E)
            keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)
            keep_mask[perm[:remove_E]] = False
            edge_index = edge_index[:, keep_mask]

        # Add (avoid self-loops; if undirected_hint, add symmetric pairs)
        add_E = remove_E
        if add_E > 0:
            add_src = self._randint_low_tri(num_nodes, add_E, device=edge_index.device)  # (k, 2)
            add_edges = add_src.t().contiguous()  # (2, add_E)
            if self.undirected_hint:
                # Duplicate reversed edges to roughly preserve undirected density
                add_edges = torch.cat([add_edges, add_edges.flip(0)], dim=1)  # (2, 2*add_E)
            edge_index = torch.cat([edge_index, add_edges], dim=1)

        new_data = data.clone()
        new_data.edge_index = edge_index
        return new_data

    def _attr_mask(self, data: Data, ratio: float) -> Data:
        if data.x is None:
            return data
        x = data.x
        N = x.size(0)
        mask_count = int(round(ratio * N))
        if mask_count <= 0:
            return data

        idx = self._randperm(N)[:mask_count]
        x_masked = x.clone()
        x_masked[idx] = 0  # node-wise masking: zero out all features for selected nodes

        new_data = data.clone()
        new_data.x = x_masked
        return new_data

    # ------------------------- Random-Walk Subgraph --------------------------

    def _subgraph_rw(self, data: Data, keep_ratio: float) -> Data:
        """
        Random-walk-based subgraph sampling, following GraphCL's RW-style subgraph view.

        Steps:
          1) Choose a random start node.
          2) Perform a random walk, accumulating unique nodes until we reach the target size.
          3) If we get stuck (isolated), restart from a random node.
        """
        x = getattr(data, "x", None)
        num_nodes = x.size(0) if x is not None else int(data.num_nodes)

        keep_count = max(1 if not self.allow_empty else 0, int(round(keep_ratio * num_nodes)))
        keep_count = min(max(keep_count, 1), num_nodes)
        if keep_count == num_nodes:
            return data

        # Build neighbor lists (optionally treat as undirected-ish for better connectivity)
        ei = data.edge_index
        device = ei.device if ei is not None else None
        if ei is None or ei.numel() == 0:
            # Degenerate graph: fallback to uniform keep
            keep_idx = self._randperm(num_nodes)[:keep_count]
            keep_idx, _ = torch.sort(keep_idx)
            return self._induce_by_nodes(data, keep_idx)

        # Construct adjacency lists
        neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
        src, dst = ei[0].tolist(), ei[1].tolist()
        for u, v in zip(src, dst):
            neighbors[u].append(v)
            if self.undirected_hint:
                neighbors[v].append(u)

        # Random-walk until we collect keep_count unique nodes
        def _randint(n: int) -> int:
            if self._rng is not None:
                return int(torch.randint(n, (1,), generator=self._rng).item())
            return int(random.randrange(n))

        visited = set()
        # start node
        current = _randint(num_nodes)
        visited.add(current)

        # walk
        # (Limit steps to avoid pathological loops on tiny/degenerate graphs.)
        max_steps = max(keep_count * 10, keep_count + 10)
        steps = 0
        while len(visited) < keep_count and steps < max_steps:
            nbrs = neighbors[current]
            if len(nbrs) == 0:
                # jump to a random node if stuck
                current = _randint(num_nodes)
                visited.add(current)
            else:
                j = _randint(len(nbrs))
                current = nbrs[j]
                visited.add(current)
            steps += 1

        # If somehow still short (e.g., many isolated nodes), pad via random picks
        if len(visited) < keep_count:
            remaining = keep_count - len(visited)
            extra = []
            # avoid duplicates
            while len(extra) < remaining:
                candidate = _randint(num_nodes)
                if candidate not in visited:
                    extra.append(candidate)
            visited.update(extra)

        keep_idx = torch.tensor(sorted(visited), dtype=torch.long, device=device)
        return self._induce_by_nodes(data, keep_idx)

    # ---- Helpers ------------------------------------------------------------

    def _induce_by_nodes(self, data: Data, keep_idx: Tensor) -> Data:
        """Induce a subgraph consisting of `keep_idx` nodes; remap edge_index and slice node attrs."""
        device = keep_idx.device
        N = int(data.num_nodes)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[keep_idx] = True

        # Remap node indices
        new_id = torch.full((N,), -1, dtype=torch.long, device=device)
        new_id[keep_idx] = torch.arange(keep_idx.size(0), device=device, dtype=torch.long)

        edge_index = data.edge_index
        if edge_index is not None and edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            e_mask = mask[src] & mask[dst]
            ei = edge_index[:, e_mask]
            ei = new_id[ei]
        else:
            ei = edge_index

        new = data.clone()

        # Slice common node attributes if present and 1st dim is N
        def _maybe_slice(attr):
            if attr is None:
                return None
            if isinstance(attr, Tensor) and attr.size(0) == N:
                return attr[keep_idx]
            return attr  # leave untouched (e.g., graph-level attrs)

        new.x = _maybe_slice(getattr(data, "x", None))
        new.pos = _maybe_slice(getattr(data, "pos", None))
        new.batch = None  # single-graph Data; Batch will set this when rebatching
        new.edge_index = ei
        new.num_nodes = int(keep_idx.size(0))
        return new

    def _randperm(self, n: int) -> Tensor:
        if self._rng is not None:
            # torch.randperm ignores device for CPU generator; move to CPU then back if needed by caller
            return torch.randperm(n, generator=self._rng)
        return torch.randperm(n)

    def _randint_low_tri(self, n: int, k: int, device=None) -> Tensor:
        """
        Sample k unordered node pairs (u, v), u != v, roughly uniform.
        Returns LongTensor of shape (k, 2) on `device`.
        """
        if k <= 0:
            return torch.empty(0, 2, dtype=torch.long, device=device)
        u = torch.randint(0, n, (k,), device=device)
        v = torch.randint(0, n - 1, (k,), device=device)
        v = v + (v >= u)  # ensure v != u
        # order (min, max) to discourage duplicates if undirected_hint=True
        a = torch.minimum(u, v)
        b = torch.maximum(u, v)
        return torch.stack([a, b], dim=1)
