import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Graph_conv_layer


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.gc1 = Graph_conv_layer(self.input_dim, self.hidden_dim)
        self.gc2 = Graph_conv_layer(self.hidden_dim, self.output_dim)
        self.dropout = dropout

    def forward(self, inputs, adj_matrix):
        x = F.relu(self.gc1.forward(inputs, adj_matrix))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2.forward(x, adj_matrix)
        return F.log_softmax(x, dim=-1)