import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data, accuracy
from model import GCN
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val):
    model.train()
    output = model.forward(features, adj)
    loss = F.nll_loss(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch:{: 4d}\tloss_train:{:.4f}\tacc_train:{:.4f}\tloss_val:{:.4f}\tacc_val:{:.4f}'.format(epoch+1, loss, acc, loss_val, acc_val))


def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model.forward(features, adj)
    loss = F.nll_loss(output[idx_test], labels[idx_test])
    acc = accuracy(output[idx_test], labels[idx_test])
    print('Test result\tloss:{:.4f}\tacc:{:.4f}'.format(loss, acc))


if __name__ == '__main__':
    seed = 2020
    hidden_dim = 16
    dropout = 0.5
    learning_rate = 0.01
    weight_decay = 5e-4
    epochs = 200

    set_seed(seed)

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = GCN(input_dim=features.shape[1], hidden_dim=hidden_dim, output_dim=labels.max().item()+1, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for e in range(epochs):
        train(e, model, optimizer, adj, features, labels, idx_train, idx_val)

    test(model, adj, features, labels, idx_test)