import dgl




# assertion
def check_key(g: dgl.DGLHeteroGraph,
            nkeys=["feat", "label", "train_mask", "val_mask", "test_mask"],
            ekeys=[]):
    for key in nkeys:
        if key not in g.ndata:
            raise KeyError(f"{key} not in graph.ndata")
    for key in ekeys:
        if key not in g.edata:
            raise KeyError(f"{key} not in graph.edata")


# data functions return a DGLHeteroGraph
def cora()->dgl.DGLHeteroGraph:
    '''
        2k node
        10k 
        7 label
        140 train
        500 valid
        1k test
    '''
    # cora only has one graph
    g = dgl.data.CoraGraphDataset(raw_dir='/home/wuyao/.dgl/cora',
                            force_reload=False,
                            verbose=False)[0]
    check_key(g)
    g.__name__ = 'cora'
    return g

def citeseer() -> dgl.DGLHeteroGraph:
    '''
        3k node
        9k edge
        6 label
        140 train
        500 valid
        1k test
    '''
    # citeseer only has one graph
    g = dgl.data.CiteseerGraphDataset(raw_dir="/home/wuyao/.dgl/citeseer",
                                    force_reload=False,
                                    verbose=False)[0]
    check_key(g)
    g.__name__ = 'citeseer'
    return g

def pubmed()->dgl.DGLHeteroGraph:
    '''
        19k node
        88k edge
        3 label
        60 train
        500 valid
        1k test
    '''
    g = dgl.data.PubmedGraphDataset(raw_dir="/home/wuyao/.dgl/pubmed",
                                    force_reload=False,
                                    verbose=False)[0]
    check_key(g)
    g.__name__ = 'pubmed'
    return g

def reddit() -> dgl.DGLHeteroGraph:
    '''
        230k node
        114m edge
        602 feat
        153k train
        22k valid
        55k test
    '''
    # reddit only has one graph
    g = dgl.data.RedditDataset(raw_dir="/home/wuyao/.dgl/reddit",
                                    force_reload=False,
                                    verbose=False)[0]
    check_key(g)
    g.__name__ = 'reddit'
    return g 



