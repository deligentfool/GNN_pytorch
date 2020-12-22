import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Graph_atten_layer


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, head_num, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.attentions = nn.ModuleList([Graph_atten_layer(self.input_dim, self.hidden_dim, self.dropout, concat=True) for _ in range(self.head_num)])
        self.out_attentions = Graph_atten_layer(self.hidden_dim * self.head_num, self.output_dim, self.dropout, concat=False)

    def forward(self, inputs, adj_matrix):
        x = F.dropout(inputs, self.dropout, training=self.training)
        x = torch.cat([atten(inputs, adj_matrix) for atten in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attentions(x, adj_matrix))
        return F.log_softmax(x, dim=-1)