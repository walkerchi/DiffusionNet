from abc import abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as gnn
import dgl.function as gfn

"""
    NN networks
"""


"""
    Parent class
"""
class Net(nn.Module):
    '''
        Template of the module
        extended module should inherit from this
    '''
    def __init__(self, in_dim:int, out_dim:int):
        super(Net, self).__init__()
        self.__in_dim = in_dim
        self.__out_dim = out_dim

    @property
    def in_dim(self):
        return self.__in_dim

    @property
    def out_dim(self):
        return self.__out_dim

    @abstractmethod
    def forward(self, g:dgl.DGLHeteroGraph)->torch.FloatTensor:
        '''
            Args:
                g, dgl.DGLHeteroGraph, assert 'feat' in g.ndata
            Returns:
                torch.FloatTensor([num_nodes, out_dim])
        '''
        pass


"""
    Sons
"""
class MLP(Net):
    def __init__(self, in_dim: int, out_dim: int,
                num_layers=2, num_hidden=128,
                batch_norm:bool=True, softmax:bool=True):

        super(MLP, self).__init__(in_dim, out_dim)
        __mlp = OrderedDict({})
        i = 0
        if num_layers <= 1:
            num_hidden = out_dim
        if batch_norm:
            __mlp[f'bn{i}'] = nn.BatchNorm1d(in_dim)
        __mlp[f'linear{i}'] = nn.Linear(in_dim, num_hidden)
        __mlp[f'relu{i}'] = nn.ReLU()
        i+=1
        for _ in range(num_layers - 2):
            if batch_norm:
                __mlp[f'bn{i}'] = nn.BatchNorm1d(num_hidden)
            __mlp[f'linear{i}'] = nn.Linear(num_hidden, num_hidden)
            __mlp[f'relu{i}'] = nn.ReLU()
            i += 1
        if num_layers > 1:
            if batch_norm:
                __mlp[f'bn{i}'] = nn.BatchNorm1d(num_hidden)
            __mlp[f'linear{i}'] = nn.Linear(num_hidden ,out_dim)
            __mlp[f'relu{i}'] = nn.ReLU()
        
        if softmax:
            __mlp['softmax'] = nn.Softmax()

        self.__mlp = nn.Sequential(__mlp)

    def forward(self, g):

        h = self.__mlp(g.ndata['feat'])

        return h

class LightGCN(Net):
    def __init__(self, in_feats:int, num_classes:int):
        super(LightGCN, self).__init__(in_feats, num_classes)
        self.conv1 = gnn.GraphConv(in_feats, num_classes)

    def forward(self, g):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, g.ndata['feat'])

        return h

class MultiGNN(Net):
    def __init__(self, in_dim:int, out_dim:int, 
                        conv, conv_arg:dict={}, 
                        hid_dim:int=128, num_hid:int=1):
        super().__init__(in_dim, out_dim)
        '''
            in_dim:  dim of feat
            out_dim: num_classes
            conv   : conv layer 
            conv_arg: extra arg for conv layer
            hid_dim
            num_hid
        '''
        assert conv in (gnn.GraphConv, gnn.SAGEConv, gnn.GATConv)

        self.self_loop = True if conv in (gnn.GraphConv, gnn.GATConv) else False

        self.__convs = nn.ModuleList([])
        while num_hid:
            num_hid -= 1
            self.__convs.append(conv(
                in_dim,
                hid_dim if num_hid else out_dim,
                **conv_arg
            ))
            in_dim = hid_dim

    def forward(self, g):
        
        g = dgl.add_self_loop(g) if self.self_loop else dgl.remove_self_loop(g)

        h = g.ndata['feat']
        for conv in self.__convs:
            h = conv(g, h)
        
        return h