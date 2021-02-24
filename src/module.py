from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as gnn
import dgl.function as gfn



class Module(nn.Module):
    '''
        Template of the module
        extended module should inherit from this
    '''
    def __init__(self, in_dim:int, out_dim:int):
        super(Module, self).__init__()
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


class MLP(Module):
    def __init__(self, in_dim, out_dim):
        super(MLP,self).__init__(in_dim, out_dim)
        self.__mlp = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,out_dim)
        )

    def forward(self,g):

        # feat = Adj . feat
        g.update_all(gfn.copy_u('feat', 'm'), gfn.sum('m', 'feat'))
        h = self.__mlp(g.ndata['feat'])

        return h

class LightGCN(Module):
    def __init__(self, in_feats, num_classes):
        super(LightGCN, self).__init__(in_feats, num_classes)
        self.conv1 = gnn.GraphConv(in_feats, num_classes)

    def forward(self, g):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, g.ndata['feat'])

        return h

class MultiGNN(Module):
    def __init__(self, in_dim, out_dim, 
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