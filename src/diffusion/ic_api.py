import torch
from torch.utils.cpp_extension import load
import dgl
import sys
import os
import time
import numpy as np
import scipy.sparse as sp

prefix = os.path.join(*os.path.split(sys.argv[0])[:-1])
cpp_module = load(name="ic_cpp",
            sources=[os.path.join(prefix,"diffusion/ic.cpp")],
            verbose=True)

def ic_slow_cpu(g:dgl.DGLHeteroGraph, 
                prob:torch.Tensor, 
                topo_change:bool=True, 
                sample_times:int=1, 
                dunbar:int=150)->dgl.DGLHeteroGraph:
    assert g.device == torch.device("cpu")

    start = time.time()

    new_edge = cpp_module.ic_slow_cpu(
                g.number_of_nodes(),
                g.edges()[0],
                g.edges()[1],
                prob,
                topo_change,
                sample_times,
                150
    )
    g = dgl.graph(tuple(new_edge), num_nodes=g.number_of_nodes())

    print(f"diffusion total consumes:{time.time() - start}")
    return g


def ic_fast_cpu(g:dgl.DGLHeteroGraph, 
                prob:torch.Tensor, 
                topo_change:bool=True, 
                sample_times:int=1, 
                dunbar:int=150):
    
    assert g.device == torch.device("cpu")

    start = time.time()

    adj = g.adjacency_matrix_scipy(transpose=True, fmt='csr')


    indptr, indices = cpp_module.ic_fast_cpu(
                adj.indptr,
                adj.indices,
                prob.numpy(),
                topo_change,
                sample_times,
                150
    )

    # print(indptr)
    # print(indices)

    new_adj = sp.csr_matrix((np.ones(indices.shape[0]), indices, indptr), shape=(g.number_of_nodes(), g.number_of_nodes()))
    g = dgl.from_scipy(new_adj)
    print(g)

    print(f"diffusion total consumes:{time.time() - start}")
    return g




