import torch
import torch.nn as nn
from core.network.block.basic import get_activation, get_norm_1d
from utils.distance import normalize, euclidean_dist, batch_euclidean_dist


def normalize_undirected_graph(A):
    """
      A: torch tensor with shape [N, V, V]
    """
    num_node = A.size(1)
    I = torch.eye(num_node).unsqueeze(0).to(A.device)
    A_ = A + I
    Dl = torch.sum(A_, dim=2) ** (-0.5)
    Dn = torch.zeros_like(A)
    for i in range(num_node):
        Dn[:, i, i] = Dl[:, i]
    DAD = torch.bmm(torch.bmm(Dn, A_), Dn)
    return DAD


class GraphicalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(GraphicalConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels * kernel_size, kernel_size=1, bias=bias)

    def forward(self, x, adj):
        """
          x: [N, C_in, V]
          y: [N, C_out, V]
          adj: [K, V, V] or [N, K, V, V]
        """
        x = self.conv(x)
        n, kc, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, v)
        if len(adj.shape) == 4:
            assert adj.size(0) == x.size(0)
            assert adj.size(1) == self.kernel_size
            x = torch.einsum('nkcv,nkvw->ncw', x, adj)
        else:
            assert adj.size(0) == self.kernel_size
            x = torch.einsum('nkcv,kvw->ncw', x, adj)
        return x.contiguous(), adj


class NodeLinearProject(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', activ='none'):
        super(NodeLinearProject, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = get_norm_1d(out_channels, norm)
        self.out = get_activation(activ)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        return self.out(x)


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm='bn', activ='relu', bias=True):
        super(GCNBlock, self).__init__()
        self.conv = GraphicalConv(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        self.norm = get_norm_1d(out_channels, norm)
        self.out = get_activation(activ)

    def forward(self, x, adj):
        x, adj = self.conv(x, adj)
        if self.norm:
            x = self.norm(x)
        x = self.out(x)
        return x, adj


class ResGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm='bn', activ='relu', residual=True):
        super(ResGCNBlock, self).__init__()
        self.conv1 = GCNBlock(in_channels, out_channels, kernel_size)
        self.conv2 = NodeLinearProject(out_channels, out_channels, norm=norm)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = NodeLinearProject(in_channels, out_channels, norm=norm)
        self.out = get_activation(activ)

    def forward(self, x, adj):
        res = self.residual(x)
        x, adj = self.conv1(x, adj)
        x = self.conv2(x)
        x = self.out(x + res)
        return x, adj


class StackedGCN(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=64, kernel_size=2, n_blk=3, norm='bn'):
        super(StackedGCN, self).__init__()
        self.res_gcn_in = ResGCNBlock(in_channels, mid_channels, kernel_size, norm=norm, activ='lrelu')
        assert n_blk >= 2
        self.layers = nn.ModuleList()
        for _ in range(n_blk - 2):
            self.layers.append(ResGCNBlock(mid_channels, mid_channels, kernel_size, norm=norm, activ='lrelu'))
        self.res_gcn_out = ResGCNBlock(mid_channels, out_channels, kernel_size, norm=norm, activ='none')
        self.kernel_size = kernel_size

    def get_affinity(self, global_feat, normalize_feature=True, re_norm=True):
        """
          global_feat: torch tensor with shape [V, D]
          adj: torch tensor with shape [1, V, V]
        """
        # normalize features
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        # euclidean similarity
        adj = euclidean_dist(global_feat, global_feat)
        # multi-kernel graphs
        if self.kernel_size == 1:
            adj = adj.unsqueeze(0)
        elif self.kernel_size == 2:
            adj = torch.stack((1-0.5*adj, 0.5*adj), dim=0)
        else:
            assert 0, 'Invalid kernel_size={} for computing affinity matrix'.format(self.kernel_size)
        # re-normalize trick
        if re_norm:
            adj = normalize_undirected_graph(adj)
        return adj

    def batch_get_affinity(self, global_feat, normalize_feature=True, re_norm=True):
        """
          global_feat: torch tensor with shape [N, V, D]
          adj: torch tensor with shape [N, V, V]
        """
        # normalize features
        if normalize_feature:
            global_feat = normalize(global_feat, dim=-1)
        # euclidean similarity
        adj = batch_euclidean_dist(global_feat, global_feat)
        # multi-kernel graphs
        if self.kernel_size == 1:
            adj = adj.unsqueeze(1)
        elif self.kernel_size == 2:
            adj = torch.stack((1-0.5*adj, 0.5*adj), dim=1)
        else:
            assert 0, 'Invalid kernel_size={} for computing affinity matrix'.format(self.kernel_size)
        # re-normalize trick
        if re_norm:
            n, k, v, w = adj.shape
            adj = normalize_undirected_graph(adj.reshape(n*k, v, w))
            adj = adj.reshape(n, k, v, w)
        return adj

    def _forward(self, x, adj):
        """
          x: [N, C_in, V]
          y: [N, C_out, V]
          adj: [N, V, V]
        """
        x, _ = self.res_gcn_in(x, adj)
        for layer in self.layers:
            x, _ = layer(x, adj)
        x, _ = self.res_gcn_out(x, adj)
        return x

    def batch_forward(self, x, adj=None):
        """
          x: [N, V, C_in]
          y: [N, V, C_out]
          adj: [N, V, V]
        """
        if adj is None:
            with torch.no_grad():
                adj = self.batch_get_affinity(x)
        return self._forward(x.permute(0, 2, 1), adj).permute(0, 2, 1)

    def forward(self, x, adj=None):
        """
          x: [V, C_in]
          y: [V, C_out]
          adj: [V, V]
        """
        if adj is None:
            with torch.no_grad():
                adj = self.get_affinity(x)
        return self._forward(x.t().unsqueeze(0), adj).squeeze(0).t()


################################
# Test
################################
if __name__ == '__main__':
    '''
        x.shape=(n, kc, v)
        n:  batch size
        kc: kernel_size * input_channels
        v:  nodes
    '''

    for ks in (1, 2):
        net = StackedGCN(128, 5, kernel_size=ks)

        inp = torch.rand(32, 128)
        out = net.forward(inp)
        print(out.shape)

        inp_batch = torch.rand(8, 32, 128)
        out_batch = net.batch_forward(inp_batch)
        print(out_batch.shape)
