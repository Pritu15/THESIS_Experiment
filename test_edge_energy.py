"""Unit tests for edge_energy.py (Step 9).

Designed to run on Kaggle (or any environment with torch, dgl, numpy,
pytest installed) - NOT executed while writing this file, per the session
constraint against running anything locally.

Run with:
    pytest test_edge_energy.py -v -s

(-s so the printed intermediate tensors are visible for hand-checking.)
"""

import numpy as np
import pytest
import torch

import dgl

from edge_energy import (
    build_symmetric_query_graph,
    compute_full_graph_degree,
    edge_energy_d1,
    edge_energy_d2,
    edge_energy_d3,
)
from edge_labels import build_supervised_edge_index
from fast_e import local_1hop_energy_lnorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bidirected_graph(num_nodes, undirected_pairs):
    """Build a DGL graph with both directions listed for each undirected pair -
    matching fast_e.py's assumption (see Step 9 report item 1) that every
    true neighbor relationship appears with the node as `dst` in some edge."""
    src, dst = [], []
    for a, b in undirected_pairs:
        src += [a, b]
        dst += [b, a]
    return dgl.graph(
        (torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)),
        num_nodes=num_nodes,
    )


def _node_edge_index(graph):
    """(dst, src) edge_index, matching fast_e.py's own docstring convention."""
    src, dst = graph.edges()
    return torch.stack([dst, src], dim=0)


def _dense_rayleigh_quotient(x, adjacency, deg_eps=1e-12, eps=1e-8):
    """x: [n] numpy 1-D signal. adjacency: [n,n] numpy 0/1 symmetric, no
    self-loops. Returns x^T L x / x^T x for L = I - D^-1/2 A D^-1/2, built
    with plain dense numpy - independent of edge_energy.py's own code, so
    it is a genuine external check, not a restatement of the implementation.
    """
    deg = adjacency.sum(axis=1)
    deg_safe = np.clip(deg, deg_eps, None)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(len(x)) - D_inv_sqrt @ adjacency @ D_inv_sqrt
    num = x @ L @ x
    den = x @ x
    return num / (den + eps)


def _find_edge_row(edge_pairs, u, v):
    pairs = edge_pairs.t().tolist()
    if [u, v] in pairs:
        return pairs.index([u, v])
    return pairs.index([v, u])


# ---------------------------------------------------------------------------
# (a) D3 vs explicit dense x^T L x / x^T x on hand-built graphs
# ---------------------------------------------------------------------------

def test_d3_path_graph_matches_dense_computation():
    # 4-node path: 0-1-2-3
    pairs = [(0, 1), (1, 2), (2, 3)]
    g = dgl.remove_self_loop(_bidirected_graph(4, pairs))
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # F=1

    edge_pairs = build_supervised_edge_index(g)
    print("path graph edge_pairs:\n", edge_pairs)

    sym_g = build_symmetric_query_graph(g)
    E_R, E_L = edge_energy_d3(features, edge_pairs, sym_g)
    print("D3 E_R (path):\n", E_R)
    print("D3 E_L (path):\n", E_L)

    # Edge (0,1): N(0)={0,1}, N(1)={0,1,2} -> N_e={0,1,2}.
    # Induced edges on {0,1,2}: (0,1),(1,2). Node order [0,1,2] -> x=[1,2,3].
    x = np.array([1.0, 2.0, 3.0])
    adjacency = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]],
        dtype=float,
    )
    expected = _dense_rayleigh_quotient(x, adjacency)
    idx = _find_edge_row(edge_pairs, 0, 1)
    print(f"edge (0,1): E_R={E_R[idx].item():.6f}, expected={expected:.6f}")
    assert E_R[idx].item() == pytest.approx(expected, abs=1e-5)
    assert E_L[idx].item() == pytest.approx(2.0 - expected, abs=1e-5)

    # Edge (1,2): N(1)={0,1,2}, N(2)={1,2,3} -> N_e={0,1,2,3} (the whole path).
    x2 = np.array([1.0, 2.0, 3.0, 4.0])
    adjacency2 = np.array(
        [[0, 1, 0, 0],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0]],
        dtype=float,
    )
    expected2 = _dense_rayleigh_quotient(x2, adjacency2)
    idx2 = _find_edge_row(edge_pairs, 1, 2)
    print(f"edge (1,2): E_R={E_R[idx2].item():.6f}, expected={expected2:.6f}")
    assert E_R[idx2].item() == pytest.approx(expected2, abs=1e-5)


def test_d3_star_graph_matches_dense_computation():
    # 5-node star: center 0, leaves 1,2,3,4. Leaf 4 is a feature outlier.
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
    g = dgl.remove_self_loop(_bidirected_graph(5, pairs))
    features = torch.tensor([[2.0], [1.0], [1.0], [1.0], [10.0]])

    edge_pairs = build_supervised_edge_index(g)
    sym_g = build_symmetric_query_graph(g)
    E_R, E_L = edge_energy_d3(features, edge_pairs, sym_g)
    print("star graph edge_pairs:\n", edge_pairs)
    print("D3 E_R (star):\n", E_R)

    # Every star edge's N_e is the whole graph: N(0) already contains every
    # leaf, so N_e = N(0) U N(leaf) = all 5 nodes regardless of which leaf.
    x = np.array([2.0, 1.0, 1.0, 1.0, 10.0])
    adjacency = np.array(
        [[0, 1, 1, 1, 1],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0]],
        dtype=float,
    )
    expected = _dense_rayleigh_quotient(x, adjacency)

    for leaf in (1, 2, 3, 4):
        idx = _find_edge_row(edge_pairs, 0, leaf)
        print(f"edge (0,{leaf}): E_R={E_R[idx].item():.6f}, expected={expected:.6f}")
        assert E_R[idx].item() == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# (b) E_R + E_L == 2 elementwise, and E_R in [0, 2]
# ---------------------------------------------------------------------------

def test_flip_identity_and_bounds_all_modes():
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]  # 4-cycle
    g = dgl.remove_self_loop(_bidirected_graph(4, pairs))
    features = torch.randn(4, 3)  # F=3, random signal

    edge_pairs = build_supervised_edge_index(g)

    node_energy_right = local_1hop_energy_lnorm(features, _node_edge_index(g))
    E_R1, E_L1 = edge_energy_d1(node_energy_right, edge_pairs)

    full_degree = compute_full_graph_degree(_node_edge_index(g), g.num_nodes())
    E_R2, E_L2 = edge_energy_d2(features, edge_pairs, full_degree)

    sym_g = build_symmetric_query_graph(g)
    E_R3, E_L3 = edge_energy_d3(features, edge_pairs, sym_g)

    for name, E_R, E_L in (("D1", E_R1, E_L1), ("D2", E_R2, E_L2), ("D3", E_R3, E_L3)):
        print(f"{name} E_R:\n{E_R}")
        print(f"{name} E_L:\n{E_L}")
        assert torch.allclose(E_R + E_L, torch.full_like(E_R, 2.0), atol=1e-5), name
        assert (E_R >= -1e-4).all(), f"{name}: E_R below 0"
        assert (E_R <= 2.0 + 1e-4).all(), f"{name}: E_R above 2"


# ---------------------------------------------------------------------------
# (c) Symmetry: E(u,v) == E(v,u)
# ---------------------------------------------------------------------------

def test_symmetry_d2_and_d3():
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    g = dgl.remove_self_loop(_bidirected_graph(4, pairs))
    features = torch.randn(4, 2)

    forward_pairs = torch.tensor([[0], [1]], dtype=torch.int64)  # edge (0,1)
    backward_pairs = torch.tensor([[1], [0]], dtype=torch.int64)  # edge (1,0)

    full_degree = compute_full_graph_degree(_node_edge_index(g), g.num_nodes())
    E_R2_fwd, _ = edge_energy_d2(features, forward_pairs, full_degree)
    E_R2_bwd, _ = edge_energy_d2(features, backward_pairs, full_degree)
    print("D2 forward (0,1):", E_R2_fwd)
    print("D2 backward (1,0):", E_R2_bwd)
    assert torch.allclose(E_R2_fwd, E_R2_bwd, atol=1e-6)

    sym_g = build_symmetric_query_graph(g)
    E_R3_fwd, _ = edge_energy_d3(features, forward_pairs, sym_g)
    E_R3_bwd, _ = edge_energy_d3(features, backward_pairs, sym_g)
    print("D3 forward (0,1):", E_R3_fwd)
    print("D3 backward (1,0):", E_R3_bwd)
    assert torch.allclose(E_R3_fwd, E_R3_bwd, atol=1e-6)


# ---------------------------------------------------------------------------
# (d) Invariance to duplicated reverse edges and to added self-loops
# ---------------------------------------------------------------------------

def test_invariance_to_duplicate_edges_and_self_loops():
    pairs = [(0, 1), (1, 2), (2, 3)]
    g_clean = dgl.remove_self_loop(_bidirected_graph(4, pairs))
    features = torch.randn(4, 2)

    edge_pairs_clean = build_supervised_edge_index(g_clean)
    sym_g_clean = build_symmetric_query_graph(g_clean)
    E_R_clean, _ = edge_energy_d3(features, edge_pairs_clean, sym_g_clean)

    # Build the same graph but with every directed edge listed twice
    # (duplicate reverse-and-forward pairs), plus self-loops on every node.
    src, dst = g_clean.edges()
    dup_src = torch.cat([src, src])
    dup_dst = torch.cat([dst, dst])
    g_dup = dgl.graph((dup_src, dup_dst), num_nodes=4)
    g_dup = dgl.add_self_loop(g_dup)

    edge_pairs_dup = build_supervised_edge_index(g_dup)
    sym_g_dup = build_symmetric_query_graph(g_dup)
    E_R_dup, _ = edge_energy_d3(features, edge_pairs_dup, sym_g_dup)

    print("edge_pairs_clean:\n", edge_pairs_clean)
    print("edge_pairs_dup:\n", edge_pairs_dup)
    assert edge_pairs_clean.shape == edge_pairs_dup.shape
    # Order may differ (torch.unique's ordering is not guaranteed identical
    # run-to-run/build-to-build), so compare as sets of (u,v,E_R-row) triples.
    clean_map = {tuple(p): E_R_clean[i].tolist() for i, p in enumerate(edge_pairs_clean.t().tolist())}
    dup_map = {tuple(p): E_R_dup[i].tolist() for i, p in enumerate(edge_pairs_dup.t().tolist())}
    assert clean_map.keys() == dup_map.keys()
    for key in clean_map:
        assert clean_map[key] == pytest.approx(dup_map[key], abs=1e-5)


# ---------------------------------------------------------------------------
# (e) Isolated / degree-1 endpoints do not produce NaN or divide-by-zero
# ---------------------------------------------------------------------------

def test_no_nan_with_isolated_and_degree_one_nodes():
    # Node 4 is isolated (no edges at all); node 0/3 have degree 1.
    pairs = [(0, 1), (1, 2), (2, 3)]
    g = dgl.remove_self_loop(_bidirected_graph(5, pairs))  # node 4 isolated
    features = torch.randn(5, 2)

    edge_pairs = build_supervised_edge_index(g)  # isolated node 4 has no incident edge

    node_energy_right = local_1hop_energy_lnorm(features, _node_edge_index(g))
    assert torch.isfinite(node_energy_right).all()

    full_degree = compute_full_graph_degree(_node_edge_index(g), g.num_nodes())
    E_R1, E_L1 = edge_energy_d1(node_energy_right, edge_pairs)
    E_R2, E_L2 = edge_energy_d2(features, edge_pairs, full_degree)

    sym_g = build_symmetric_query_graph(g)
    E_R3, E_L3 = edge_energy_d3(features, edge_pairs, sym_g)

    for name, E_R, E_L in (("D1", E_R1, E_L1), ("D2", E_R2, E_L2), ("D3", E_R3, E_L3)):
        print(f"{name} E_R (isolated-node graph):\n{E_R}")
        assert torch.isfinite(E_R).all(), f"{name}: non-finite E_R"
        assert torch.isfinite(E_L).all(), f"{name}: non-finite E_L"


# ---------------------------------------------------------------------------
# (f) Constant signal gives E_R = 0, E_L = 2
# ---------------------------------------------------------------------------

def test_constant_signal_gives_zero_right_energy():
    # A constant signal only lands in the normalized Laplacian's null space
    # when every node has EQUAL degree (the null eigenvector is D^1/2 . 1,
    # not the flat vector 1, on an irregular graph - e.g. a star graph's
    # center has degree 4 vs leaf degree 1, and a literal constant signal
    # there gives num[center] = 4*(c/sqrt(4) - c/sqrt(1))^2 != 0). Use a
    # 4-cycle instead: every node has degree 2, so this holds exactly.
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle, 2-regular
    g = dgl.remove_self_loop(_bidirected_graph(4, pairs))
    features = torch.full((4, 2), 3.0)  # constant signal, F=2

    node_energy_right = local_1hop_energy_lnorm(features, _node_edge_index(g))
    print("node-level E_R (constant signal):\n", node_energy_right)
    assert torch.allclose(node_energy_right, torch.zeros_like(node_energy_right), atol=1e-5)

    edge_pairs = build_supervised_edge_index(g)
    full_degree = compute_full_graph_degree(_node_edge_index(g), g.num_nodes())
    E_R2, E_L2 = edge_energy_d2(features, edge_pairs, full_degree)
    print("D2 E_R (constant signal):\n", E_R2)
    assert torch.allclose(E_R2, torch.zeros_like(E_R2), atol=1e-5)
    assert torch.allclose(E_L2, torch.full_like(E_L2, 2.0), atol=1e-5)

    sym_g = build_symmetric_query_graph(g)
    E_R3, E_L3 = edge_energy_d3(features, edge_pairs, sym_g)
    print("D3 E_R (constant signal):\n", E_R3)
    assert torch.allclose(E_R3, torch.zeros_like(E_R3), atol=1e-5)
    assert torch.allclose(E_L3, torch.full_like(E_L3, 2.0), atol=1e-5)


# ---------------------------------------------------------------------------
# (g) EGNN camouflage construction: E_R(e) decreases after camouflaging
# ---------------------------------------------------------------------------

def test_camouflage_decreases_right_energy():
    # Star: center 0, leaves 1,2,3,4. Leaf 4 starts as an outlier (10.0);
    # camouflage it by replacing its feature with the mean of its neighbors
    # (its only neighbor is the center, 0) - mimicking the EGNN paper's
    # construction (Section 2.2: replacing a node's features with its
    # neighbors' average feature value).
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
    g = dgl.remove_self_loop(_bidirected_graph(5, pairs))

    features_before = torch.tensor([[2.0], [1.0], [1.0], [1.0], [10.0]])
    edge_pairs = build_supervised_edge_index(g)
    sym_g = build_symmetric_query_graph(g)

    E_R_before, _ = edge_energy_d3(features_before, edge_pairs, sym_g)
    idx = _find_edge_row(edge_pairs, 0, 4)
    print(f"Before camouflage: E_R(0,4) = {E_R_before[idx].item():.6f}")

    # Camouflage node 4: replace its feature with the mean of its neighbors'
    # features (its only neighbor is node 0, whose feature is 2.0).
    features_after = features_before.clone()
    features_after[4, 0] = features_before[0, 0]  # mean of {2.0} = 2.0

    E_R_after, _ = edge_energy_d3(features_after, edge_pairs, sym_g)
    print(f"After camouflage:  E_R(0,4) = {E_R_after[idx].item():.6f}")

    assert E_R_after[idx].item() < E_R_before[idx].item()
