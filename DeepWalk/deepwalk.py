import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# todo: no implement of Hierarchical Softmax

class DeepWalk(nn.Module):
    def __init__(self, node_num, hidden_dim):
        super(DeepWalk, self).__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(self.node_num, self.hidden_dim, bias=False)
        self.output_layer = nn.Linear(self.hidden_dim, self.node_num)

    def forward(self, input):
        return F.log_softmax(self.output_layer(self.input_layer(input)), dim=-1)

    def show_embedding(self, input):
        return self.input_layer(input)


def random_walk(node_adjlist, start_node, max_length):
    node_list = [start_node]
    current_node = start_node
    for _ in range(max_length):
        current_node = random.choice(node_adjlist[current_node])
        node_list.append(current_node)
    return node_list


def skip_gram(node_list, window_size, model, node_num, optimizer):
    for idx, node in enumerate(node_list):
        for adj_idx in range(max(idx - window_size, 0), min(idx + window_size, len(node_list))):
            one_hot = torch.zeros(1, node_num).float()
            one_hot[0, node_list[adj_idx]] = 1

            out = model.forward(one_hot)
            target = torch.tensor(node_list[idx]).long().unsqueeze(0)
            loss = F.nll_loss(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    node_list = [0, 1, 2, 3, 4, 5, 6, 7]
    node_adjlist = [[1, 2, 3], [0, 2, 3, 7], [0, 1, 3], [0, 1, 2, 7], [5, 6], [4, 6], [4, 5], [1, 3]]
    # * i-th element in node_adjlist means the i-th node's neighbor node
    node_num = 8
    hidden_dim = 2
    learning_rate = 0.025
    window_size = 3
    max_length = 6
    max_epoch = 200

    model = DeepWalk(node_num, hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(max_epoch):
        random.shuffle(node_list)
        for node in node_list:
            node_history = random_walk(node_adjlist, node, max_length)
            skip_gram(node_history, window_size, model, node_num, optimizer)

    one_hots = torch.eye(node_num)
    embedding = model.show_embedding(one_hots)
    print(embedding)