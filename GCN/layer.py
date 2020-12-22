import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Graph_conv_layer(nn.Module):
    def __init__(self, input_dim, output_dim, is_bias=True):
        super(Graph_conv_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_bias = is_bias

        self.weight = nn.parameter.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        if self.is_bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_matrix):
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj_matrix, support)
        if self.bias is not None:
            output = output + self.bias
        return output