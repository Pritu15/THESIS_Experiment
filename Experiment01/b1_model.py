"""B1: GraphSAGE with the existing right spectral-energy signal."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class B1RightEnergySAGE(nn.Module):
    """Early-fusion GraphSAGE over raw features and normalized right energy."""

    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_classes,
        dropout=0.5,
        aggregator_type="mean",
    ):
        super().__init__()
        fused_dim = 2 * in_feats
        self.conv1 = SAGEConv(fused_dim, hidden_dim, aggregator_type)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2, aggregator_type)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def forward(self, graph, features, energy_right):
        """Return node logits from X concatenated with normalized E_R.

        Args:
            graph: DGL graph used by each GraphSAGE layer.
            features: Original node features with shape [N, F].
            energy_right: Training-statistic-normalized E_R with shape [N, F].
        """
        fused_input = torch.cat([features, energy_right], dim=1)

        h = self.conv1(graph, fused_input)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)
