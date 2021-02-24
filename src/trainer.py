import os
import time
import scipy.sparse as sp
import sklearn.metrics as M
import torch
import dgl


from module import Module, MLP,LightGCN, MultiGNN
from graph import GraphProcessor, IC


class Trainer:
    datasets = {
        'cora':
        dgl.data.CoraGraphDataset(raw_dir='../Data',
                                force_reload=False,
                                verbose=False)
    }

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
                g,
                model,
                processor=None,
                loss_fn = 'ce',
                metric_fn = 'f1-mic',
                buf:bool  = False
                ):
        assert isinstance(g, (str, dgl.DGLHeteroGraph))
        if isinstance(g, str):
            assert g in self.datasets.keys()
            g_name = g
            g = self.datasets[g][0]
        else:
            g_name = 'cache'
        assert 'feat' in g.ndata
        assert 'label' in g.ndata

        assert processor is None or issubclass(processor, GraphProcessor)
        assert issubclass(type(model), Module) or issubclass(model, Module)
        
        assert 'train_mask' in g.ndata, f"{g.ndata}"
        assert 'val_mask' in g.ndata, f"{g.ndata}"
        assert 'test_mask'  in g.ndata, f"{g.ndata}"

        assert loss_fn in self.loss_fns.keys()
        assert metric_fn in self.metric_fns.keys()


        if buf and self.has_cache(g_name):
            g = self.load_graph(g_name)
        else:
            if processor is not None:
                g = processor.process(g)
            if buf:
                self.save_graph(g, g_name)

        self.g = {}
        self.g['train'] = dgl.node_subgraph(g, g.ndata['train_mask'])
        self.g['valid'] = dgl.node_subgraph(g, g.ndata['val_mask'])
        self.g['test']  = dgl.node_subgraph(g, g.ndata['test_mask'])
        
        if issubclass(model, Module):
            self.model = model(int(g.ndata['feat'].size(-1)), int(g.ndata['label'].max() - g.ndata['label'].min() + 1))
        else:
            self.model = model
        
        

        self.loss_fn = self.loss_fns[loss_fn]
        self.metric_fn = self.metric_fns[metric_fn]

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


    def train(self, lr:float=1e-3, epoch:int=100, test:bool=True):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
    
    trainer = Trainer(
        g='cora',
        processor=IC,
        model=LightGCN
    ) 
    trainer.train()