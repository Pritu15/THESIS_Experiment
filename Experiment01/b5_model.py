"""B5: edge-level GraphSAGE with gated, edge-centered bidirectional
spectral energy (Step 9).

The encoder (three SAGEConv layers, identical configuration to B0/B4) is a
separately-instantiated copy, matching the convention already established
by every other arm in this repository (B0-B4 each declare their own
encoder layers rather than sharing weight objects across arms).

Fusion has two modes:
  "gated"        - G_e = sigmoid(MLP([h_u || h_v])); a per-feature gate
                   computed from ENCODER EMBEDDINGS, not from the energy
                   values themselves. Zen_e = G_e * proj_L(E_L) +
                   (1 - G_e) * proj_R(E_R), with SEPARATE learnable
                   projections per branch. This is what breaks the
                   E_L = 2 - E_R redundancy (see the Step 9 report): once
                   proj_L != proj_R and the gate depends on information
                   orthogonal to the energy itself, [proj_R(E_R), proj_L(E_L)]
                   no longer collapses to a fixed affine function of E_R.
  "naive_concat" - Zen_e = [E_R || E_L] with no gate and no projections.
                   Included so the redundancy claim can be checked
                   empirically: this mode is expected to perform ~identically
                   to using E_R alone, since E_L is an exact affine function
                   of E_R and a linear classifier gains nothing from it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class EdgeEnergyGate(nn.Module):
    """Gate computed from encoder embeddings [h_u || h_v], not from the
    energy branches - the mixing weight is a function of information
    orthogonal to the values being mixed."""

    def __init__(self, embed_dim, energy_dim, hidden_dim=16):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, energy_dim),
            nn.Sigmoid(),
        )
        # Same convention as E_train.SpectralFlipGatingMLP: start biased
        # toward the flipped (E_L) branch.
        with torch.no_grad():
            self.gate_mlp[-2].weight.zero_()
            self.gate_mlp[-2].bias.fill_(-2.0)

    def forward(self, h_u, h_v):
        return self.gate_mlp(torch.cat([h_u, h_v], dim=1))


class B5EdgeBidirectionalEnergySAGE(nn.Module):
    """GraphSAGE encoder (B0/B4-identical configuration) + edge-centered
    bidirectional spectral energy, gated by encoder embeddings."""

    def __init__(
        self,
        in_feats,
        energy_dim,
        hidden_dim,
        num_classes,
        dropout=0.5,
        aggregator_type="mean",
        gate_hidden_dim=16,
        fusion="gated",
    ):
        super().__init__()
        if fusion not in ("gated", "naive_concat"):
            raise ValueError(f"Unknown fusion mode: {fusion!r}")
        self.fusion = fusion
        embed_dim = hidden_dim // 2

        # Encoder: identical shape/activation/dropout to B0GraphSAGE / B4.
        self.conv1 = SAGEConv(in_feats, hidden_dim, aggregator_type)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type)
        self.conv3 = SAGEConv(hidden_dim, embed_dim, aggregator_type)
        self.dropout = dropout

        if fusion == "gated":
            self.gate = EdgeEnergyGate(embed_dim, energy_dim, gate_hidden_dim)
            self.proj_R = nn.Linear(energy_dim, energy_dim)
            self.proj_L = nn.Linear(energy_dim, energy_dim)
            zen_dim = energy_dim
        else:  # naive_concat
            self.gate = None
            self.proj_R = None
            self.proj_L = None
            zen_dim = 2 * energy_dim

        classifier_in = 2 * embed_dim + zen_dim
        self.edge_head = nn.Sequential(
            nn.Linear(classifier_in, embed_dim),
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

    def forward(self, graph, features, edge_pairs, energy_right, energy_left):
        """Return (edge_logits, gate_or_None, fused_energy).

        Args:
            graph: full graph, same object passed to B0-B4.
            features: [N, in_feats].
            edge_pairs: LongTensor [2, M] canonical (u, v) pairs.
            energy_right, energy_left: [M, energy_dim] precomputed, frozen,
                train-stat-normalized edge energy (from edge_energy.py).
        """
        h = self.encode(graph, features)  # [N, embed_dim]
        h_u = h[edge_pairs[0]]  # [M, embed_dim]
        h_v = h[edge_pairs[1]]  # [M, embed_dim]

        if self.fusion == "gated":
            gate = self.gate(h_u, h_v)  # [M, energy_dim]
            zen = gate * self.proj_L(energy_left) + (1.0 - gate) * self.proj_R(energy_right)
        else:  # naive_concat
            gate = None
            zen = torch.cat([energy_right, energy_left], dim=1)  # [M, 2*energy_dim]

        z_e = torch.cat([h_u, h_v, zen], dim=1)
        logits = self.edge_head(z_e)
        return logits, gate, zen
