import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph_atten_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, concat):
        super(Graph_atten_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.concat = concat

        self.w = nn.parameter.Parameter(torch.empty(size=(self.input_dim, self.output_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * self.output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, inputs, adj_matrix):
        wh = torch.mm(inputs, self.w)
        # * wh: [N, output_dim]

        N = wh.size(0)
        wh_repeat = wh.repeat([N, 1])
        wh_alter_repeat = wh.repeat_interleave(repeats=N, dim=0)

        atten_input = torch.cat([wh_alter_repeat, wh_repeat], dim=1).view(N, N, -1)
        # * atten_input: [N, N, 2 * output_dim]
        attention = F.leaky_relu(torch.matmul(atten_input, self.a).squeeze(-1))
        # * attention: [N, N]
        mask = -torch.ones_like(attention) * 9e15
        attention = torch.where(adj_matrix.to_dense() > 0, attention, mask)

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        h_prime = torch.matmul(attention, wh)
        # * h_prime: [N, output_dim]

        if self.concat:
            return F.leaky_relu(h_prime)
        else:
            return h_prime
