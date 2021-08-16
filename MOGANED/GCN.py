"""
MOGANED model for event extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MOGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, K, dropout, device, alpha=0.2):
        super(MOGCN,self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.in_drop = dropout
        self.K = K
        self.layers_a = nn.ModuleList()
        self.layers_b = nn.ModuleList()
        self.layers_c = nn.ModuleList()
        for i in range(self.K):
            self.layers_a.append(GraphAttentionLayer(in_features=in_dim,
                            out_features=hidden_dim, dropout=dropout,
                        alpha=alpha, device=device, concat=False))
            self.layers_b.append(GraphAttentionLayer(in_features=in_dim,
                                                     out_features=hidden_dim, dropout=dropout,
                                                     alpha=alpha, device=device, concat=False))
            self.layers_c.append(GraphAttentionLayer(in_features=in_dim,
                                                     out_features=hidden_dim, dropout=dropout,
                                                     alpha=alpha, device=device, concat=False))

        self.Wawa = nn.Sequential(
            nn.Linear(self.hidden_dim, 100),
            nn.Tanh(),
        )
        self.Ctx = nn.Linear(100, 1, bias=False)


    def forward(self, adj, inputs):
        adj_a, adj_b, adj_c = adj[:, 0, :, :], adj[:, 1, :, :], adj[:, 2, :, :]
        layer_list = []
        for i in range(self.K):
            gat_a = self.layers_a[i](matmuls(adj_a,i), inputs)
            gat_b = self.layers_b[i](matmuls(adj_b, i), inputs)
            gat_c = self.layers_c[i](matmuls(adj_c, i), inputs)
            outputs = gat_a + gat_b + gat_c
            layer_list.append(outputs)

        s_ctxs = []
        for i in range(self.K):
            s_raw = self.Wawa(layer_list[i])
            ctx_apply = self.Ctx(s_raw)
            s_ctxs.append(ctx_apply)
        vs = F.softmax(torch.cat(s_ctxs, dim=2), dim=2)  # [Batch,maxlen,3]

        h_concats = torch.cat([torch.unsqueeze(layer_list[layer], 2) for layer in range(self.K)], dim=2)
        final_h = torch.sum(torch.mul(torch.unsqueeze(vs, 3), h_concats), dim=2)

        return final_h


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, device, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, adj, input):
        h = torch.matmul(input, self.W)  # [B,N,D]
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)  # [B,N,N,2D]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B,N,N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)  # [B,N,N]
        h_prime = torch.matmul(attention, h)  # [B,N,D]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

def matmuls(a,times):
    res = a
    for i in range(times):
        res = torch.matmul(res,a)
    return res


