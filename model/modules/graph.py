import math

import torch
from torch import nn

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}

# [추가] 대칭 관절 연결
SYM_CONNECTIONS = { 
    1: [4], 4: [1], # 양쪽 엉덩이 (Hip) 
    2: [5], 5: [2], # 양쪽 무릎 (Knee) 
    3: [6], 6: [3], # 양쪽 발목 (Ankle) 
    11: [14], 14: [11], # 양쪽 어깨 (Shoulder) 
    12: [15], 15: [12], # 양쪽 팔꿈치 (Elbow) 
    13: [16], 16: [13] # 양쪽 손목 (Wrist) 
} 

# [추가] 코어-말단 관절 연결
CORE_CONNECTIONS = { 
    0: [3, 6, 13, 16], # 골반 <-> 양 발목, 양 손목 
    8: [3, 6, 13, 16], # 목/상체 중심 <-> 양 발목, 양 손목 
    3: [0, 8], 6: [0, 8], 13: [0, 8], 16: [0, 8] 
}

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None):
        self.nodes_ = """
        :param dim_int: Channel input dimension
        :param dim_out: Channel output dimension
        :param num_nodes: Number of nodes
        :param neighbour_num: Neighbor numbers. Used in temporal GCN to create edges
        :param mode: Either 'spatial' or 'temporal'
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param connections: Spatial connections for graph edges (Optional)
        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        self._init_gcn()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()

            #가중치
            self.W3 = nn.Linear(dim_in, dim_out, bias=False)
            self.W4 = nn.Linear(dim_in, dim_out, bias=False)

            #행렬 저장
            sym_adj = self._init_relation_adj(SYM_CONNECTIONS)
            core_adj = self._init_relation_adj(CORE_CONNECTIONS)

            self.register_buffer("sym_adj", sym_adj, persistent=False)
            self.register_buffer("core_adj", core_adj, persistent=False)

            #sigmoid 계산 시 0인데 0.5로 계산되는 것을 막는다(관절 간 연결이 없는데도 가중치가 추가 연산되는 것을 방지함)
            relation_mask = (sym_adj.any(dim=1) | core_adj.any(dim=1))
            self.register_buffer(
                "relation_mask", relation_mask[:, None], persistent=False
            )
        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)

    def _init_gcn(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj

    def _init_relation_adj(self, connections):
        adj = torch.zeros(self.num_nodes, self.num_nodes)

        for joint, neighbors in connections.items():
            adj[joint, neighbors] = 1

        degree_inv_sqrt = adj.sum(dim=1).clamp_min(1).rsqrt()
        return degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :] #얼마나 영향을 받는지 계수를 계산한다

    def _init_temporal_adj(self, connection_length):
        """Connects each joint to itself and the same joint withing next `connection_length` frames."""
        adj = torch.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            try:
                for j in range(connection_length + 1):
                    adj[i, i + j] = 1
            except IndexError:  # next j frame does not exist
                pass
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape

        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.eye(n)
        dev = adj.get_device()
        if dev >= 0:
            norm_deg_matrix = norm_deg_matrix.to(dev)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)

        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        b, t, j, c = x.shape
        if self.mode == 'temporal':
            x = x.transpose(1, 2)  # (B, T, J, C) -> (B, J, T, C)
            x = x.reshape(-1, t, c)
            if self.use_temporal_similarity:
                similarity = x @ x.transpose(1, 2)
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                adj = (similarity >= threshold).float()
            else:
                adj = self.adj
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * j, 1, 1)

        else:
            x = x.reshape(-1, j, c)
            adj = self.adj
            adj = self.change_adj_device_to_cuda(adj)
            adj = adj.repeat(b * t, 1, 1)

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)

        if self.mode == 'spatial': #대칭, 코어-말단으로 연결된 특징 가져와서 시그모이드
            #대칭관절 / 코어-말단관절 가중치를 어느 정도 반영할지, 0 < 반영안함
            SYM_weight = 1
            CORE_weight = 1

            sym_features = self.sym_adj @ x
            core_features = self.core_adj @ x

            relation_logits = SYM_weight * self.W3(sym_features) + CORE_weight * self.W4(core_features)
            relation_update = torch.sigmoid(relation_logits) * self.relation_mask

        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))

        if self.mode == 'spatial':
            x = x + relation_update #새로 만든 특징 더해주기
            x = x.reshape(b, t, j, self.dim_out)
        else:
            x = x.reshape(b, j, t, self.dim_out).transpose(1, 2)

        return x
