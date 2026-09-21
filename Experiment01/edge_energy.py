"""Edge-centered right/left spectral energy - Step 9.

Three candidate definitions of an edge-level E_R(e), for edge e=(u,v):

  D1 (endpoint average):    0.5 * (E_R(u) + E_R(v))
       using the EXISTING, UNMODIFIED node-level fast_e.local_1hop_energy_lnorm.
       A deterministic function of two already-available node-level values -
       cannot demonstrate that edge-centered computation adds anything (see
       the Step 9 report).

  D2 (single-edge term):    (x_u/sqrt(d_u) - x_v/sqrt(d_v))^2 / (x_u^2/d_u + x_v^2/d_v)
       using each endpoint's FULL-GRAPH degree. Cheapest, but has no
       neighborhood context beyond the two endpoints.

  D3 (induced-subgraph Rayleigh quotient) on N_e = N(u) U N(v), where N(i)
       includes i itself (matching the existing node-level convention).
       Degrees are computed WITHIN the induced subgraph, not from the full
       graph - this is required for the result to be a valid classical
       Rayleigh quotient (eigenvalues of the induced normalized Laplacian
       bounded in [0,2]); using full-graph degrees here would not produce a
       real normalized Laplacian of the induced subgraph. This was an
       explicit, approved design decision (Step 9, Phase 9A item 3), not
       inherited silently from fast_e.py's own full-graph-degree shortcut.

None of this modifies fast_e.py. D1 calls the existing node-level function
unchanged; D2 duplicates a ~3-line degree computation (fast_e.py never
exposes its internal degree tensor, and must not be edited to do so).

All three modes return (E_R, E_L) with shape [M, F] each, where M is the
number of supervised edges from edge_labels.build_supervised_edge_index
(canonical, deduplicated, self-loop-free, u < v), and E_L = 2 - E_R exactly
(same flip identity as fast_e.local_1hop_energy_lnorm_flip).
"""

from typing import Optional, Tuple

import dgl
import torch

from edge_labels import build_supervised_edge_index


def _clamp_and_flip(E_R: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    return E_R, 2.0 - E_R


@torch.no_grad()
def compute_full_graph_degree(
    edge_index: torch.Tensor, num_nodes: int, deg_eps: float = 1e-12
) -> torch.Tensor:
    """In-degree per node, matching fast_e.py's own convention (row 0 = dst).

    Duplicated here in three lines rather than imported, because
    fast_e.local_1hop_energy_lnorm computes this internally and never
    returns it, and must not be modified to do so.
    """
    dst = edge_index[0]
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    return deg.clamp_min(deg_eps)


@torch.no_grad()
def edge_energy_d1(
    node_energy_right: torch.Tensor, edge_pairs: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """D1: endpoint average of the existing node-level E_R."""
    u, v = edge_pairs[0], edge_pairs[1]
    E_R = 0.5 * (node_energy_right[u] + node_energy_right[v])
    return _clamp_and_flip(E_R, eps)


@torch.no_grad()
def edge_energy_d2(
    features: torch.Tensor,
    edge_pairs: torch.Tensor,
    full_degree: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """D2: single-edge Rayleigh term over the two endpoints' full-graph degrees."""
    u, v = edge_pairs[0], edge_pairs[1]
    inv_sqrt_deg = full_degree.rsqrt()
    inv_deg = full_degree.reciprocal()

    xi = features[u] * inv_sqrt_deg[u].unsqueeze(1)
    xj = features[v] * inv_sqrt_deg[v].unsqueeze(1)
    num = (xi - xj).square()

    den = (
        features[u].square() * inv_deg[u].unsqueeze(1)
        + features[v].square() * inv_deg[v].unsqueeze(1)
    )
    E_R = num / (den + eps)
    return _clamp_and_flip(E_R, eps)


@torch.no_grad()
def build_symmetric_query_graph(graph: "dgl.DGLGraph") -> "dgl.DGLGraph":
    """Undirected, simple, self-loop-free graph used only for D3's
    neighbor/topology queries - independent of whichever directed or
    undirected form is fed to the GNN encoder elsewhere. This is a
    separate, throwaway copy built once per graph; it does not modify or
    replace the graph object the encoder consumes.

    `to_simple` collapses any parallel/duplicate edges before
    `to_bidirected` (which requires a simple graph as input) - relevant
    because Amazon/Yelp are homogenized from multi-relation graphs and can
    contain duplicate (u, v) pairs (see the Phase 8A audit).

    `graph` may be on GPU (b5_train.py moves the encoder's graph to
    `device` before this is ever called), but `dgl.to_simple` only
    supports CPU graphs. Moved to CPU here rather than by the caller,
    since this function already documents itself as building a separate,
    throwaway copy independent of whatever form the encoder consumes.
    """
    g = graph.to("cpu")
    g = dgl.remove_self_loop(g)
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g)
    return g


@torch.no_grad()
def edge_energy_d3(
    features: torch.Tensor,
    edge_pairs: torch.Tensor,
    symmetric_query_graph: "dgl.DGLGraph",
    eps: float = 1e-8,
    deg_eps: float = 1e-12,
    verbose_every: int = 200_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """D3: classical Rayleigh quotient of the induced subgraph on
    N_e = N(u) U N(v), with degrees computed WITHIN the subgraph.

    N(i) = {i} union its neighbors under `symmetric_query_graph` (built by
    build_symmetric_query_graph), independent of whatever directed/
    undirected form the GNN encoder itself consumes.

    Cost: this calls dgl.khop_in_subgraph once per supervised edge - O(M)
    Python-level calls, each DGL-native rather than a hand-rolled
    adjacency-list walk, but with no batched/vectorized subgraph
    extraction. For datasets with millions of supervised edges (e.g. full
    Amazon, M ~ 4.4M) this loop is the dominant cost and has not been
    benchmarked at that scale - see --d3_edge_limit in b5_train.py.
    """
    device = features.device
    M = edge_pairs.shape[1]
    F = features.shape[1]
    E_R = torch.zeros(M, F, device=device, dtype=features.dtype)

    u_list = edge_pairs[0].tolist()
    v_list = edge_pairs[1].tolist()

    for idx in range(M):
        u, v = u_list[idx], v_list[idx]
        seeds = torch.tensor([u, v], dtype=torch.int64, device=symmetric_query_graph.device)
        sg, _ = dgl.khop_in_subgraph(symmetric_query_graph, seeds, k=1)

        node_ids = sg.ndata[dgl.NID].to(device)
        x_sub = features[node_ids]

        # to_bidirected => in_degree == undirected degree within N_e (each
        # distinct neighbor contributes exactly one incoming arc).
        deg_sub = sg.in_degrees().to(device=device, dtype=features.dtype).clamp_min(deg_eps)
        inv_sqrt_deg = deg_sub.rsqrt()
        inv_deg = deg_sub.reciprocal()

        # Deduplicate to unique undirected pairs so each induced edge is
        # counted exactly once in the numerator (sg is bidirected, so
        # sg.edges() alone would double-count every edge). sg is CPU (it
        # comes from the now-CPU symmetric_query_graph), so this returns
        # CPU indices - moved to `device` before indexing x_sub/inv_*_deg,
        # which live on `device` alongside the rest of the model's tensors.
        sub_edge_pairs = build_supervised_edge_index(sg).to(device)
        if sub_edge_pairs.shape[1] == 0:
            # (u, v) itself is always an edge of the induced subgraph, so
            # this should not happen; guarded rather than left as 0/0.
            continue
        p_idx, q_idx = sub_edge_pairs[0], sub_edge_pairs[1]

        xi = x_sub[p_idx] * inv_sqrt_deg[p_idx].unsqueeze(1)
        xj = x_sub[q_idx] * inv_sqrt_deg[q_idx].unsqueeze(1)
        num_f = (xi - xj).square().sum(dim=0)
        den_f = (x_sub.square() * inv_deg.unsqueeze(1)).sum(dim=0)
        E_R[idx] = num_f / (den_f + eps)

        if verbose_every and (idx + 1) % verbose_every == 0:
            print(f"  D3 energy: {idx + 1:,}/{M:,} edges processed")

    return _clamp_and_flip(E_R, eps)


def compute_edge_energy(
    mode: str,
    features: torch.Tensor,
    edge_pairs: torch.Tensor,
    graph: "dgl.DGLGraph",
    node_energy_right: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to D1 / D2 / D3 by name. Returns (E_R, E_L), each [M, F]."""
    if mode == "d1":
        if node_energy_right is None:
            raise ValueError("D1 requires node_energy_right (the existing node-level E_R).")
        return edge_energy_d1(node_energy_right, edge_pairs, eps=eps)

    if mode == "d2":
        src, dst = graph.edges()
        edge_index = torch.stack([dst, src], dim=0)
        full_degree = compute_full_graph_degree(edge_index, graph.num_nodes())
        return edge_energy_d2(features, edge_pairs, full_degree, eps=eps)

    if mode == "d3":
        symmetric_query_graph = build_symmetric_query_graph(graph)
        return edge_energy_d3(features, edge_pairs, symmetric_query_graph, eps=eps)

    raise ValueError(f"Unknown edge energy mode: {mode!r} (expected 'd1', 'd2', or 'd3')")
