import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import GraphSAGE
from utils import load_data
import random


if __name__ == '__main__':
    add_self = False
    concat = True
    aggregator_method = 'mean'
    hidden_dim = 128
    sample_num = 5
    learning_rate = 0.7
    max_epoch = 100

    feat_data, labels, adj_list = load_data()
    input_dim = feat_data.shape[1]
    output_dim = len(set(labels))
    graphsage = GraphSAGE(add_self, concat, aggregator_method, feat_data, input_dim, output_dim, hidden_dim, adj_list, sample_num)

    rand_indices = np.random.permutation(len(feat_data))
    test_node = rand_indices[:1000]
    valid_node = rand_indices[1000: 1500]
    train_node = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(graphsage.parameters(), lr=learning_rate)

    for epoch in range(max_epoch):
        batch_nodes = train_node[: 256]
        random.shuffle(batch_nodes)
        batch_nodes = torch.LongTensor(batch_nodes)
        pred = graphsage.forward(batch_nodes).log()
        target = torch.LongTensor(labels[batch_nodes])
        loss = F.nll_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid_node = torch.LongTensor(valid_node)
        valid_pred = graphsage.forward(valid_node).max(-1)[1]
        acc = (valid_pred == torch.LongTensor(labels[valid_node])).sum().item() / len(valid_node)
        print('epoch:{}\tacc:{:.4f}'.format(epoch + 1, acc))
