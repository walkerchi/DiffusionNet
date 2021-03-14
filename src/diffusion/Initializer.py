from abc import abstractmethod
import torch
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
    def __call__(self, g: dgl.DGLHeteroGraph, key: str, dtype)->dgl.DGLHeteroGraph:
        return g



"""
    Sons
"""
class Constant(NodeInitializer):
    def __init__(self, val=1):
        self.val = val
    def __call__(self, g: dgl.DGLHeteroGraph, key: str, dtype=torch.float32) -> dgl.DGLHeteroGraph:
        '''
            initialize the node attr[key] with constant val
            key should be new 
        '''
        NodeInitializer.check_key(g, key)
        g.ndata[key] = torch.full((g.number_of_nodes,), self.val, dtype=dtype)
        return g

class Eye(NodeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph, key: str, dtype=torch.bool) -> dgl.DGLHeteroGraph:
        '''
            initialize the ndoe att[key] with a eye matrix `I'
            key should be new
        '''
        NodeInitializer.check_key(g, key)
        g.ndata[key] = torch.full((g.number_of_nodes(), g.number_of_nodes()), True, dtype=dtype)
        return g









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
    def __call__(self,g: dgl.DGLHeteroGraph, key: str, dtype) -> dgl.DGLHeteroGraph:
        pass



"""
    Sons
    inherit from parent class:EdgeInitailizer
"""

class Constant(EdgeInitializer):
    def __init__(self, val=1):
        super(Constant, self).__init__()
        self.val = val
    def __call__(self, g: dgl.DGLHeteroGraph, key: str, dtype=torch.float32) -> dgl.DGLHeteroGraph:
        '''
            initialize the edge attr[key]  with constant val
            key should be new
        '''
        EdgeInitializer.check_key(g, key)
        g.edata[key] = torch.full((g.number_of_edges(),), self.val, dtype=dtype)
        return g


class OutDegree(EdgeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph, key: str, dtype=torch.float32) -> dgl.DGLHeteroGraph:
        '''
            initialize the edge attr[key] with out_degree
            key should be new
        '''
        EdgeInitializer.check_key(g, key)
        NodeInitializer.check_key(g, "__out_degree")
        g.ndata["__out_degree"] = g.out_degrees().type(dtype)
        def message_fn(edges):
            edges.data[key] = edges.src["__out_degree"]
        g.apply_edges(message_fn)
        g.ndata.pop("__out_degree")
        return g


class InDegree(EdgeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph, key: str, dtype=torch.float32) -> dgl.DGLHeteroGraph:
        '''
            initialize the edge attr[key] with in_degree
            key should be new
        '''
        EdgeInitializer.check_key(g, key)
        NodeInitializer.check_key(g, "__in_degree")
        g.ndata["__in_degree"] = g.in_degrees().type(dtype)
        def message_fn(edges):
            edges.data[key] = edges.dst["__in_degree"]
        g.apply_edges(message_fn)
        g.ndata.pop("__in_degree")
        return g


class Degree(EdgeInitializer):
    def __call__(self, g: dgl.DGLHeteroGraph, key: str, dtype=torch.float32) -> dgl.DGLHeteroGraph:
        '''
            initialize the edge attr[key] with degree
            key should be new
        '''
        EdgeInitializer.check_key(g, key)
        NodeInitializer.check_key(g, "__in_degree")
        NodeInitializer.check_key(g, "__out_degreee")
        g.ndata["__in_degree"] = g.in_degrees().type(dtype)
        g.ndata["__out_degree"] = g.out_degrees().type(dtype)
        def message_fn(edges):
            edges.data[key] = edges.src["__out_degree"] + edges.dst["__in_degree"]
        g.apply_edges(message_fn)
        g.ndata.pop("__in_degree")
        g.ndata.pop("__out_degree")
        return g