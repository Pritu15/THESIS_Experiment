"""B4: static edge-level GraphSAGE baseline (no spectral energy).

The encoder (three SAGEConv layers: in_feats -> hidden_dim -> hidden_dim ->
hidden_dim // 2, ReLU + dropout after each) is a separately-instantiated
copy of B0GraphSAGE's encoder (b0_model.py), with identical layer count,
dimensions, aggregator, activation, and dropout placement, so that B4's
node representations are produced by "the same encoder configuration" as
the static node experiments. It is a separate class (not an import of
B0GraphSAGE) only because B0GraphSAGE returns final classifier logits and
never exposes the pre-classifier node embedding that edge classification
needs; b0_model.py itself is left untouched.

Edge scoring follows the simplest possible design requested for Step 8:
    h_e = [h_u || h_v]  ->  MLP  ->  edge anomaly logits
No edge spectral energy, no graph-level pooling, no temporal component.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class B4EdgeAnomalySAGE(nn.Module):
    """GraphSAGE node encoder (B0-identical configuration) + edge-pair MLP head."""

    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_classes,
        dropout=0.5,
        aggregator_type="mean",
    ):
        super().__init__()
        # Encoder: identical shape/activation/dropout to B0GraphSAGE.
        self.conv1 = SAGEConv(in_feats, hidden_dim, aggregator_type)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2, aggregator_type)
        self.dropout = dropout

        # Edge head: simplest concat-then-MLP design.
        embed_dim = hidden_dim // 2
        self.edge_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def encode(self, graph, features):
        """Return node embeddings with shape [num_nodes, hidden_dim // 2]."""
        h = self.conv1(graph, features)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, graph, features, edge_pairs):
        """Return edge logits with shape [num_supervised_edges, num_classes].

        Args:
            graph: DGL graph used by the GraphSAGE encoder (full graph,
                same object passed to B0-B3 - message passing is
                transductive over every node regardless of which edges
                are being scored).
            features: Node features, [N, in_feats].
            edge_pairs: LongTensor [2, M] canonical (u, v) pairs from
                edge_labels.build_supervised_edge_index.
        """
        h = self.encode(graph, features)  # [N, hidden_dim // 2]
        h_u = h[edge_pairs[0]]  # [M, hidden_dim // 2]
        h_v = h[edge_pairs[1]]  # [M, hidden_dim // 2]
        h_e = torch.cat([h_u, h_v], dim=1)  # [M, 2 * hidden_dim // 2]
        return self.edge_head(h_e)  # [M, num_classes]
