import dgl




def check_key(g: dgl.DGLHeteroGraph,
            nkeys=["feat", "label", "train_mask", "val_mask", "test_mask"],
            ekeys=[]):
    for key in nkeys:
        if key not in g.ndata:
            raise KeyError(f"{key} not in graph.ndata")
    for key in ekeys:
        if key not in g.edata:
            raise KeyError(f"{key} not in graph.edata")

def cora()->dgl.DGLHeteroGraph:
    g = dgl.data.CoraGraphDataset(raw_dir='../Data',
                            force_reload=False,
                            verbose=False)[0]
    check_key(g)
    g.__name__ = 'cora'
    return g




dataset = {
    'cora':cora()
    
}