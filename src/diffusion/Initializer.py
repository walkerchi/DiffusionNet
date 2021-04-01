from abc import abstractmethod
import torch
import numpy as np
import dgl
import dgl.function as gfn




#####################################################################
#                       Node Initializer                            #
#####################################################################
"""
    NodeInitailizer is to init the value of the attribute of the node
    To customize the function, inherit from NodeInitializer's __call__
"""

"""
    Parent class
"""
class NodeInitializer:
    def __init__(self):
        pass

    @staticmethod
    def check_key(g: dgl.DGLHeteroGraph, key: str):
        if key in g.ndata:
            raise KeyError(f"{key} is already in the g.ndata, change the key!")

    @abstractmethod
    def __call__(self, g: dgl.DGLHeteroGraph, dtype)->torch.tensor:
        pass



"""
    Sons
"""
class NodeConstant(NodeInitializer):
    def __init__(self, val=1):
        self.val = val

    def __call__(self, g: dgl.DGLHeteroGraph, dtype=torch.float32) -> torch.tensor:
        '''
            initialize the node attr[key] with constant val
            key should be new 
        '''
        # NodeInitializer.check_key(g, key)
        return torch.full((g.number_of_nodes,), self.val, dtype=dtype)

class Eye(NodeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph, dtype=torch.bool) -> torch.tensor:
        '''
            initialize the node att[key] with a eye matrix `I'
            key should be new
        '''
        # NodeInitializer.check_key(g, key)
        return torch.eye(g.number_of_nodes(), dtype=dtype)

class ParitialEye(NodeInitializer):

    def __call__(self, base:int, batch_size:int, g:dgl.DGLHeteroGraph, dtype=torch.bool) -> torch.tensor:
        '''
            initialize the node att[key] with parital eye from base to base + batch_size [num_node, batch_size]
            key should be new
        '''
        # NodeInitializer.check_key(g, key)
        th = torch.zeros(g.number_of_nodes(),batch_size)
        th[(torch.arange(base, base + batch_size), torch.arange(batch_size))] = 1
        return th.type(dtype)








#####################################################################
#                       Edge Initializer                            #
#####################################################################

"""
    Edge initializer is to init the value of attributes of the edge
    To customize the initializer, you could inherit from the EdgeInitializer
"""
"""
    Parent class
"""
class EdgeInitializer:
    def __init__(self):
        pass

    @staticmethod
    def check_key(g:dgl.DGLHeteroGraph, key:str):
        if key in g.edata:
            raise KeyError(f"{key} is already in the g.edata, change the key!")

    @abstractmethod
    def __call__(self,g: dgl.DGLHeteroGraph, dtype) -> torch.tensor:
        pass



"""
    Sons
    inherit from parent class:EdgeInitailizer
"""

class EdgeConstant(EdgeInitializer):
    def __init__(self, val=1):
        super(EdgeConstant, self).__init__()
        self.val = val

    def __call__(self, g: dgl.DGLHeteroGraph, dtype=torch.float32) -> torch.tensor:
        '''
            initialize the edge attr[key]  with constant val
            key should be new
        '''
        # EdgeInitializer.check_key(g, key)
        return torch.full((g.number_of_edges(),), self.val, dtype=dtype)


class OutDegree(EdgeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph, dtype=torch.float32) -> torch.tensor:
        '''
            initialize the edge attr[key] with out_degree
            key should be new
        '''
        # EdgeInitializer.check_key(g, key)
        NodeInitializer.check_key(g, "__out_degree")
        g.ndata["__out_degree"] = g.out_degrees().type(dtype).to(g.device)

        def message_fn(edges):
            edges.data["__out_degree"] = edges.src["__out_degree"]
        g.apply_edges(message_fn)
        g.ndata.pop("__out_degree")
        th = g.edata['__out_degree']
        g.edata.pop("__out_degree")
        return th


class InDegree(EdgeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph, dtype=torch.float32) -> torch.tensor:
        '''
            initialize the edge attr[key] with in_degree
            key should be new
        '''
        # EdgeInitializer.check_key(g, key)
        NodeInitializer.check_key(g, "__in_degree")
        g.ndata["__in_degree"] = g.in_degrees().type(dtype).to(g.device)

        def message_fn(edges):
            edges.data['__in_degree'] = edges.dst["__in_degree"]
        g.apply_edges(message_fn)
        g.ndata.pop("__in_degree")
        th = g.edata['__in_degree']
        g.edata.pop("__in_degree")
        return th


class Degree(EdgeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph,  dtype=torch.float32) -> dgl.DGLHeteroGraph:
        '''
            initialize the edge attr[key] with degree
            key should be new
        '''
        # EdgeInitializer.check_key(g, key)
        NodeInitializer.check_key(g, "__in_degree")
        NodeInitializer.check_key(g, "__out_degreee")
        g.ndata["__in_degree"] = g.in_degrees().type(dtype)
        g.ndata["__out_degree"] = g.out_degrees().type(dtype)
        def message_fn(edges):
            edges.data["__degree"] = edges.src["__out_degree"] + edges.dst["__in_degree"]
        g.apply_edges(message_fn)
        g.ndata.pop("__in_degree")
        g.ndata.pop("__out_degree")
        th = g.edata["__degree"]
        g.edata.pop("__degree")
        return th