import numpy as np


def load_data(path="./data/cora/", dataset="cora"):
    feat_data = np.genfromtxt("{}{}.content".format(path, dataset), dtype=str)
    cite_data = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    label_list = list(set(feat_data[:, -1]))

    label_map = dict(zip(label_list, list(range(len(label_list)))))
    node_map = dict(zip(feat_data[:, 0].astype(np.int64), list(range(len(feat_data)))))
    labels = np.zeros(len(feat_data), dtype=np.int64)
    for k, v in label_map.items():
        labels[feat_data[:, -1] == k] = v
    feat_data = feat_data[:, 1: -1].astype(np.float32)

    adj_list = {}
    for i in range(len(cite_data)):
        v1, v2 = node_map[cite_data[i, 0]], node_map[cite_data[i, 1]]
        if v1 not in adj_list.keys():
            adj_list[v1] = []
        adj_list[v1].append(v2)
        if v2 not in adj_list.keys():
            adj_list[v2] = []
        adj_list[v2].append(v1)
    return feat_data, labels, adj_list