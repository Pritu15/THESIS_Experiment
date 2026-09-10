"""B3: GraphSAGE with adaptively gated right and left spectral energy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

from E_train import SpectralFlipGatingMLP


class B3BidirectionalEnergySAGE(nn.Module):
    """Reuse EGNN's gate, then early-fuse its energy output with raw features."""

    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_classes,
        dropout=0.5,
        aggregator_type="mean",
        gate_hidden_dim=16,
        gate_type="per_feature",
        gate_num_layers=2,
    ):
        super().__init__()
        self.gate_type = gate_type
        self.spectral_gate = SpectralFlipGatingMLP(
            in_feats=in_feats,
            hidden_dim=gate_hidden_dim,
            gate_type=gate_type,
            num_layers=gate_num_layers,
        )

        fused_dim = 2 * in_feats
        self.conv1 = SAGEConv(fused_dim, hidden_dim, aggregator_type)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2, aggregator_type)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def forward(self, graph, features, features_for_gate, energy_right, energy_left):
        """Return node logits, gate values, and gated bidirectional energy."""
        mixed_energy, gates = self.spectral_gate(
            features_for_gate,
            energy_right,
            energy_left,
        )
        fused_input = torch.cat([features, mixed_energy], dim=1)

        h = self.conv1(graph, fused_input)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)
        return logits, gates, mixed_energy
