"""B0: non-spectral GraphSAGE baseline for static node anomaly detection."""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class B0GraphSAGE(nn.Module):
    """Three-layer GraphSAGE encoder and node-level classification head.

    B0 deliberately consumes the original node features directly. It keeps the
    GraphSAGE depth, dimensions, activation, dropout, and classifier layout used
    by GatedEnergySAGE, while excluding every spectral-energy component.
    """

    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_classes,
        dropout=0.5,
        aggregator_type="mean",
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_dim, aggregator_type)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2, aggregator_type)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def forward(self, graph, features):
        """Return two-class node logits with shape [num_nodes, num_classes]."""
        h = self.conv1(graph, features)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)
