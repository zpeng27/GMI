import torch
import torch.nn as nn
from utils import process

# Applies mean-pooling on neighbors
class AvgNeighbor(nn.Module):
    def __init__(self):
        super(AvgNeighbor, self).__init__()

    def forward(self, seq, adj_ori):
        adj_ori = process.sparse_mx_to_torch_sparse_tensor(adj_ori)
        if torch.cuda.is_available():
            adj_ori = adj_ori.cuda()
        return torch.unsqueeze(torch.spmm(adj_ori, torch.squeeze(seq, 0)), 0)
