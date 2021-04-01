# %%
import os
import time
import scipy.sparse as sp
import sklearn.metrics as M
import torch
import torch.nn as nn
import dgl
import dgl.nn as gnn

import data as D
import net as N
from diffusion import Processor as P
from diffusion import Initializer as Init
# %%

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
                model:nn.Module = N.MLP,
                processor:P.GraphProcessor = None,
                loss_fn:str = 'ce',
                metric_fn:str = 'f1-mic',
                buf:bool = False,
                device:torch.device = torch.device('cpu')
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
        print("begin\n")
        self.__check_key(g)
        assert processor is None or issubclass(processor.__class__, P.GraphProcessor)
        assert issubclass(model.__class__,N.Net)

        assert loss_fn in self.loss_fns.keys()
        assert metric_fn in self.metric_fns.keys()

        # define record dir
        self.result_dir = g.__name__ + '_reuslt'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        # check cached
        if buf and self.has_cache(g.__name__) and processor is not None:
            g = self.load_graph(f"{processor.__name__}_{g.__name__}")
        else:
            g = g.to(device)
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
        self.model = model.to(device)
        self.result_txt = self.model.__name__ + '.txt'
        # loss and metric
        self.loss_fn = self.loss_fns[loss_fn]
        self.metric_fn = self.metric_fns[metric_fn]
        print("finished\n")

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

    def has_cache(self, prefix:str = ''):
        return os.path.exists(os.path.join(self.BUF_PATH, f"{prefix}_graph.bin"))

    def load_graph(self, prefix:str = ''):
        assert os.path.exists(os.path.join(self.BUF_PATH, f"{prefix}_graph.bin"))
        return dgl.load_graphs(
            os.path.join(self.BUF_PATH, f"{prefix}_graph.bin")
        )[0][0]

    def train(self, lr:float = 1e-3, epoch:int = 100, weight_decay=0, test:bool = True):

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
        with open(os.path.join(self.result_dir,self.result_txt),'w') as f:
            f.write("total time for training:"+ str(end - start) + '\n')
            f.close()
        if test:
            self.test()

    def test(self):
        self.model.eval()
        start = time.time()
        y_pred = self.model(self.g['test'])
        end = time.time()

        metric = self.metric_fn(y_pred, self.g['test'].ndata['label'])
        print(f"test  f1-mic:{metric}, reference time:{end-start}")
        with open(os.path.join(self.result_dir,self.result_txt),'a') as f:
            f.write(f"test  f1-mic:{metric}, reference time:{end-start}\n")
            f.close()  
# %%
if __name__ == '__main__':

    #
    # modify here
    #
    g = D.reddit()
    num_feats = g.ndata['feat'].size(-1)
    num_classes = (g.ndata['label'].max() - g.ndata['label'].min() + 1).item()

    print(f"Using {g.__name__}, feat:{num_feats},  class:{num_classes}")
# %%
    trainer = Trainer(
        g=g,

        # processor=P.HawkesIC(diffuse=False, hawkes_decay=0.99),
        processor=P.IC_cpp(
            probability_init=Init.EdgeConstant(0.5),
            sample_times=1,
            aggregate=True,
            verbose=True),
        # processor=None,
        model=N.MLP(num_feats, num_classes, num_layers=2, num_hidden=128,batch_norm=True, softmax=False),
        # model=N.LightGCN(num_feats, num_classes),
        # model=N.MultiGNN(num_feats, num_classes,
        #         num_layers=1, conv=gnn.SAGEConv,
        #         conv_arg={"aggregator_type":"gcn"}),
        # model=N.MultiGNN(num_feats, num_classes,
        #           num_layers=2, conv=gnn.GraphConv,
        #           conv_arg={}),
        buf=False,
        device=torch.device('cuda:0')
    )
# %%
    trainer.train(
        lr=1e-3,
        weight_decay=5e-2,
        epoch=200)
# %%
