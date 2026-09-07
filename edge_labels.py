"""Derived edge-anomaly labels for the Step 8 static edge-level extension.

None of the datasets in this repository ship a genuine per-edge anomaly
label (verified against get_amazon.py, get_yelp.py, get_tfinance.py,
get_tsocial.py: only `ndata['feature']` and `ndata['label']` are ever read
or written; `edata` is never touched). This module implements the UniGAD
(NeurIPS 2024) derived-label protocol instead, which builds an edge label
purely from the two endpoint node labels:

    P_anom(i, j) = avg(P_anom(i), P_anom(j))

For binary ground-truth node labels in {0, 1}, this average lands in
{0.0, 0.5, 1.0} depending on how many endpoints are anomalous. This module
exposes that continuous value ("soft" label) and a thresholded binary
version ("hard" label, positive iff at least one endpoint is anomalous)
for use as a classification target. Because the label is a deterministic
function of the two endpoint labels, any edge classifier built on it can,
in principle, win by rediscovering node-level information rather than
learning anything edge-intrinsic; see run_static_edge_ablation.py / the
Step 8 report for the leakage diagnostic this motivates.

This module never modifies graph structure or node data in place; it only
reads `graph.edges()` and takes label/mask tensors as plain arguments.
"""

from typing import NamedTuple

import torch


class EdgeSplit(NamedTuple):
    """Boolean membership masks over the canonical supervised edge set."""

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    excluded: torch.Tensor  # mixed-membership or touches an unlabeled node


def build_supervised_edge_index(graph: "dgl.DGLGraph") -> torch.Tensor:
    """Build the canonical, deduplicated, self-loop-free edge set for edge-level supervision.

    The loaders in this repository (get_amazon.py, get_yelp.py,
    get_tfinance.py, get_tsocial.py) all call `dgl.add_self_loop`, and
    Amazon/YelpChi are homogenized from a multi-relation graph without a
    subsequent `dgl.to_simple`, so `graph.edges()` can contain self-loops
    and parallel/duplicate node pairs (see the Phase 8A audit). Neither is
    a meaningful "edge" for anomaly classification, so both are removed
    here. Direction is also collapsed: edge (u, v) and (v, u) are treated
    as the same supervised edge, canonicalized as (min(u, v), max(u, v)).
    This is independent of whether the graph passed to the GNN encoder is
    directed or `--undirected`; the encoder still runs message passing on
    `graph` exactly as loaded.

    Args:
        graph: DGL graph as produced by one of the existing static loaders.

    Returns:
        edge_pairs: LongTensor [2, M], row 0 = u, row 1 = v, u < v, with no
            duplicate (u, v) pairs and no self-loops.
    """
    src, dst = graph.edges()
    keep = src != dst
    src, dst = src[keep], dst[keep]

    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    pairs = torch.stack([u, v], dim=1)
    unique_pairs = torch.unique(pairs, dim=0)
    return unique_pairs.t().contiguous()


def derive_edge_labels(labels: torch.Tensor, edge_pairs: torch.Tensor):
    """Compute the UniGAD-style derived soft and hard edge labels.

    Args:
        labels: LongTensor [N], ground-truth per-node anomaly label (0/1).
        edge_pairs: LongTensor [2, M] as returned by build_supervised_edge_index.

    Returns:
        soft_label: FloatTensor [M], avg(label_u, label_v) in {0.0, 0.5, 1.0}.
        hard_label: LongTensor [M], 1 if soft_label > 0 else 0 (i.e. positive
            iff at least one endpoint is anomalous).
    """
    u, v = edge_pairs[0], edge_pairs[1]
    label_u = labels[u].float()
    label_v = labels[v].float()
    soft_label = (label_u + label_v) / 2.0
    hard_label = (soft_label > 0).long()
    return soft_label, hard_label


def endpoint_stratum(labels: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
    """Classify each edge by its endpoint-label configuration.

    Returns:
        LongTensor [M] with values:
            0 = both endpoints normal
            1 = exactly one endpoint anomalous
            2 = both endpoints anomalous
    """
    u, v = edge_pairs[0], edge_pairs[1]
    label_u = labels[u].long()
    label_v = labels[v].long()
    return label_u + label_v  # 0, 1, or 2 - identical encoding to the stratum id


STRATUM_NAMES = {0: "both_normal", 1: "one_anomalous", 2: "both_anomalous"}


def assign_edge_splits(
    edge_pairs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> EdgeSplit:
    """Assign each supervised edge to train/val/test by strict endpoint membership.

    An edge is a train edge iff BOTH endpoints are train nodes, a val edge
    iff BOTH are val nodes, a test edge iff BOTH are test nodes. Any edge
    whose two endpoints fall in different node splits (or touch a node
    that is in none of the three masks, e.g. Amazon's unlabeled nodes) is
    excluded from all three supervised sets rather than silently assigned
    somewhere.

    This directly guarantees the requirement "no test edge has both
    endpoints seen with labels during training": a train-train edge can
    never be classified as a test edge under this rule, because
    train_mask and test_mask are disjoint by construction (see
    get_amazon.py / get_yelp.py / get_tfinance.py / get_tsocial.py
    `_create_splits`).

    This does NOT eliminate transductive leakage through message passing:
    the GNN encoder (b4_model.py) runs SAGEConv over the *entire* graph on
    every forward pass, exactly like B0-B3 do for node classification, so
    a test node's embedding is computed using aggregated *features* from
    its train-node neighbors (never their *labels*, which are never used
    as model input anywhere in this repository). This is the same
    transductive setting already accepted for the node-level arms; it is
    called out explicitly here, per the Step 8 instructions, rather than
    silently assumed away for edges.
    """
    u, v = edge_pairs[0], edge_pairs[1]

    both_train = train_mask[u] & train_mask[v]
    both_val = val_mask[u] & val_mask[v]
    both_test = test_mask[u] & test_mask[v]
    excluded = ~(both_train | both_val | both_test)

    return EdgeSplit(train=both_train, val=both_val, test=both_test, excluded=excluded)
