import os
import time
import scipy.sparse as sp
import sklearn.metrics as M
import torch
import torch.nn as nn
import dgl
import dgl.nn as gnn

from data import dataset
import net as N
from diffusion import Processor  as P
from diffusion import Initializer as Init



class Trainer:


    loss_fns = {
        'ce':torch.nn.CrossEntropyLoss()
    }

    metric_fns={
        'f1-mic':lambda y_pred, y_true: M.f1_score(
        y_true.clone().detach().cpu().numpy(),
        y_pred.clone().detach().cpu().argmax(-1).numpy(),
        average='micro')
    }

    BUF_PATH = '../Trainer-buffer'

    def __init__(self,
                g: dgl.DGLHeteroGraph,
                model:nn.Module=N.MLP,
                processor:P.GraphProcessor=None,
                loss_fn:str = 'ce',
                metric_fn:str='f1-mic',
                buf:bool  = False
                ):
        """
            g:              dgl.DGLHeteroGraph
                    ndata: 'feat', 'label', 'train_mask', 'val_mask', 'test_mask'
                    edata: not required
            model:          a class inherit from Net
                    or an instace of this class 
            processor:       a class inherit form GraphProcessor
            loss_fn:        str, should be in self.loss_fns.keys
            metric_fn:      str, should be in self.metric_fns.keys
            buf:            bool, whether to buffer the processed graph
        """
        self.__check_key(g)
        assert processor is None or issubclass(processor.__class__, P.GraphProcessor)
        assert issubclass(model.__class__,N.Net)

        assert loss_fn in self.loss_fns.keys()
        assert metric_fn in self.metric_fns.keys()

        # check cached
        if buf and self.has_cache(g.__name__) and processor is not None:
            g = self.load_graph(f"{processor.__name__}_{g.__name__}")
        else:
            if processor is not None:
                # if cached skip the process
                g = processor(g)
            if buf and processor is not None:
                # if need buf, cache the processed graph
                self.save_graph(g, f"{processor.__name__}_{g.__name__}")

        # split the batch into train, valid, test
        self.g = {}
        self.g['train'] = dgl.node_subgraph(g, g.ndata['train_mask'])
        self.g['valid'] = dgl.node_subgraph(g, g.ndata['val_mask'])
        self.g['test'] = dgl.node_subgraph(g, g.ndata['test_mask'])
        
        # initialize nn model
        self.model = model
        
        # loss and metric
        self.loss_fn = self.loss_fns[loss_fn]
        self.metric_fn = self.metric_fns[metric_fn]

    def __check_key(self, g: dgl.DGLHeteroGraph,
                nkeys=["feat", "label", "train_mask", "val_mask", "test_mask"],
                ekeys=[]):
        for key in nkeys:
            if key not in g.ndata:
                raise KeyError(f"{key} not in graph.ndata")
        for key in ekeys:
            if key not in g.edata:
                raise KeyError(f"{key} not in graph.edata")

        assert hasattr(g, '__name__'),"graph doesn't have a name"

    def save_graph(self, g:dgl.DGLHeteroGraph, prefix:str=''):
        if not os.path.exists(self.BUF_PATH):
            os.mkdir(self.BUF_PATH)
        dgl.save_graphs(
            os.path.join(self.BUF_PATH, f"{prefix}_graph.bin"),
            [g],
            {'glabel':torch.tensor([1])}
            )

    def has_cache(self, prefix:str=''):
        return os.path.exists(os.path.join(self.BUF_PATH, f"{prefix}_graph.bin"))

    def load_graph(self, prefix:str=''):
        assert os.path.exists(os.path.join(self.BUF_PATH, f"{prefix}_graph.bin"))
        return dgl.load_graphs(
            os.path.join(self.BUF_PATH, f"{prefix}_graph.bin")
        )[0][0]

    def train(self, lr:float=1e-3, epoch:int=100, weight_decay=0, test:bool=True):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.model.train()
        start = time.time()
        for ep in range(epoch):
            optimizer.zero_grad()

            y_pred = self.model(self.g['train'])

            loss = self.loss_fn(y_pred, self.g['train'].ndata['label'])
            metric = self.metric_fn(y_pred, self.g['train'].ndata['label'])

            loss.backward()
            optimizer.step()

            y_pred = self.model(self.g['valid'])

            valid_metric = self.metric_fn(y_pred, self.g['valid'].ndata['label'])
            print(
                f"epoch:{ep} ,loss:{loss.clone().item()}, f1-mic:{metric} , valid f1-mic:{valid_metric}"
            )
        end = time.time()
        print("total time for training:",end - start)

        if test:
            self.test()

    def test(self):
        self.model.eval()

        start = time.time()
        y_pred = self.model(self.g['test'])
        end = time.time()

        metric = self.metric_fn(y_pred, self.g['test'].ndata['label'])
        print(f"test  f1-mic:{metric}, reference time:{end-start}")  



if __name__ == '__main__':
    
    g = dataset['cora']
    num_feats = g.ndata['feat'].size(-1)
    num_classes = (g.ndata['label'].max() - g.ndata['label'].min() + 1).item()
    
    trainer = Trainer(
        g=g,

        # processor=P.HawkesIC(diffuse=False, hawkes_decay=0.99),
        processor=P.IC( activated_init=Init.Eye(),
                        probability_init=Init.Constant(0.5),
                        sample_times=5,
                        aggregate=True),
        # processor = None,

        model = N.MLP(num_feats, num_classes, num_layers=2, num_hidden=128,batch_norm=True, softmax=True)
        # model = N.LightGCN(num_feats, num_classes)
        # model= N.MultiGNN(num_feats, num_classes,
        #         num_layers=1, conv=gnn.SAGEConv,
        #         conv_arg={"aggregator_type":"mean"})
        # model=N.MultiGNN(num_feats, num_classes,
        #           num_layers=2, conv=gnn.GraphConv,
        #           conv_arg={})
    ) 
    trainer.train(
            lr=0.001,
            weight_decay=0.01,
            epoch=100)