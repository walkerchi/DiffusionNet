

import sklearn.metrics as M
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import torch
import torch.nn as nn
import dgl
import dgl.nn as gnn

datasets = {
    'cora':dgl.data.CoraGraphDataset(raw_dir='../../../DataSet', force_reload=False, verbose=False),
    'cora-full':dgl.data.CoraFullDataset(raw_dir='../../../DataSet',force_reload=False, verbose=False),
    'cite':dgl.data.CitationGraphDataset(raw_dir='../../../DataSet',force_reload=False, verbose=False),
    'citeseer':dgl.data.CiteseerGraphDataset(raw_dir='../../../DataSet',force_reload=False, verbose=False)
    }


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP,self).__init__()
        self.__mlp = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,out_dim)
        )

    def forward(self,g):
        x = self.__mlp(g.ndata['feat'])
        return x
class GraphSAGE(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(GCN,self).__init__()
        self.__gcn = nn.ModuleList([
            gnn.SAGEConv(in_dim,128,aggregator_type='mean',activation=nn.ReLU()),
            gnn.SAGEConv(128,out_dim,aggregator_type='mean')
        ])
    def forward(self, g):
        g = dgl.add_self_loop(g)
        feat = g.ndata['feat']
        for i in self.__gcn:
            feat = i(g, feat)
        return feat
class LightGCN(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(LightGCN, self).__init__()
        self.conv1 = gnn.GraphConv(in_feats, num_classes)

    def forward(self, g):
        g = dgl.add_self_loop(g)
        feat = g.ndata['feat']
        h = self.conv1(g, feat)
        return h

def train(dataset='cora', model=LightGCN, use_ic=True, lr=0.001,weight_decay=0.01, epoch=100):
    dataset = datasets[dataset]
    g = dataset[0]
    if use_ic:
        g = IC.run(g)
    g_train = dgl.node_subgraph(g, g.ndata['train_mask'])
    g_valid = dgl.node_subgraph(g, g.ndata['val_mask'])
    g_test  = dgl.node_subgraph(g, g.ndata['test_mask'])

    model = model(g.ndata['feat'].size(-1),32)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    metric_fn = lambda y_pred, y_true: M.f1_score(y_true.clone().detach().cpu().numpy(), y_pred.clone().detach().cpu().argmax(-1).numpy(), average='micro')
    for ep in range(epoch):
        optimizer.zero_grad()
        y_pred = model(g_train)
        loss = loss_fn(y_pred,g_train.ndata['label'])
        metric = metric_fn(y_pred, g_train.ndata['label'])
        loss.backward()
        optimizer.step()
        y_pred = model(g_valid)
        valid_metric = metric_fn(y_pred, g_valid.ndata['label'])
        print(f"epoch:{ep} ,loss:{loss.clone().item()}, f1-mic:{metric} , valid f1-mic:{valid_metric}")

    model.eval()
    y_pred = model(g_test)
    metric = metric_fn(y_pred, g_test.ndata['label'])
    print(f"test  f1-mic:{metric}")


class IC:
    def __init__(self):
        pass 

    @staticmethod
    def independent_cascade(g:dgl.DGLHeteroGraph,
                    activated_key:str='IC_activated',
                    probability_key:str='IC_probability',
                    stop_situation:str='dunbar',
                    sample_times:int = 2,
                    default_probability=0.5
                    )->dgl.DGLHeteroGraph:
        '''
            Args:
                g, dgl.DGLHeteroGraph   
                    g.ndata[activated_key] torch.boolTensor([i])
                    g.edata[probability_key] torch.FloatTensor([1 or i])
                activated_key, str    , default: 'IC_activated'
                probability_key, str  , default: 'IC_probability'
                stop_situation,str  in ['dunbar', 'iteration']  , default: 'dunbar'
                sample_times, int, default: 1

                
            Returns:
                g, dgl.DGLHeteroGraph
                    g.ndata[activated_key]
                    g.edata[probability_key]
        '''
        DUNBAR_NUMBER = 16
        NUM_ITER = 4


        # Assertion 
        def get_parallel_num(th:torch.Tensor)->int:
            if th.dim() == 1:
                return 1
            else:
                assert th.dim() == 2
                return th.size(1)

        assert stop_situation in ('dunbar','iteration')
        assert default_probability <= 1 and default_probability >=0
        assert sample_times > 0

        if activated_key not in g.ndata.keys():
            raise ValueError
        g.ndata[activated_key] = g.ndata[activated_key].bool()
        parallel_num = get_parallel_num(g.ndata[activated_key])


        if probability_key not in g.edata.keys():
            print("set default probability")
            g.edata[probability_key] = torch.full((g.number_of_edges(),),default_probability,dtype=torch.float32)
        g.edata[probability_key] = g.edata[probability_key].float()
        assert (g.edata[probability_key] <= 1).all()
        if g.edata[probability_key].dim() == 1 and parallel_num > 1:
            g.edata[probability_key] = g.edata[probability_key][:,None].repeat(1, parallel_num)
        else:
            assert parallel_num == get_parallel_num(g.edata[probability_key])

        assert 'IC_current_activated' not in g.ndata
        g.ndata['IC_current_activated'] = g.ndata[activated_key].clone()  #[num_nodes, num_parallel]

        # Message Passing 

        def IC_message(edges):
            
            return {'m':edges.src['IC_current_activated'].float() * edges.data[probability_key]}

        def IC_reduce(nodes):
            # nodes.mailbox['m']  [num_batch, num_neighbor, num_parallel]
            # stop_mask           [num_parallel]
            updated_activated = torch.full_like(nodes.mailbox['m'][:,0,:], False, dtype=torch.bool) #[num_batch,num_parallel]
            for _ in range(sample_times):
                sample = ( torch.rand_like(nodes.mailbox['m']) < nodes.mailbox['m']).any(1) #[num_batch, num_parallel]
                updated_activated = updated_activated | sample
            cur_activated = updated_activated & (~nodes.data[activated_key])
            activated = updated_activated | nodes.data[activated_key]
            return {activated_key:activated,'IC_current_activated':cur_activated}

        if stop_situation == 'dunbar':
            iteration = 0
            #IC.vis_activated(g,activated_key=None, show=False, save_dir=f'tmp_{g.edata[probability_key][0][0]:.1f}_{sample_times}', prefix=f"{iteration}")
            while (~g.ndata['IC_current_activated']).all():
                iteration += 1
                g.update_all(IC_message, IC_reduce)
                g.ndata['IC_current_activated'].T[torch.where(g.ndata[activated_key].int().sum(0)>=DUNBAR_NUMBER)] = False
                #IC.vis_activated(g,activated_key=None, show=False, save_dir=f'tmp_{g.edata[probability_key][0][0]:.1f}_{sample_times}', prefix=f"{iteration}")
                print(f"iteration:{iteration}")  
        elif stop_situation == 'iteration':
            for _ in range(NUM_ITER):
                g.update_all(IC_message, IC_reduce)

        g.ndata.pop('IC_current_activated')
        return g

    @staticmethod
    def activated_graph(g:dgl.DGLHeteroGraph,activated_key:str='IC_activated')->dgl.DGLHeteroGraph:
        '''
            Build a new graph from the activated_key(torch.BoolTensor([num_nodes, num_nodes])) as adj matrix
            Args:
                g, dgl.DGLHeteroGraph  input graph, 
                activated_key, str    
            Returns:
                dgl.DGLHeteroGraph
        '''
        activated = g.ndata[activated_key]
        assert activated.shape == (g.number_of_nodes(), g.number_of_nodes()),f"activated shape: {activated.shape}  node num:{g.number_of_nodes()}"
        activated_g = dgl.graph(torch.where(activated.bool()), num_nodes=g.number_of_nodes())

        for k, v in g.ndata.items():
            if k != activated_key:      
                activated_g.ndata[k] = v
           
        return activated_g

    @staticmethod
    def vis_activated(g:dgl.DGLHeteroGraph, 
                    activated_key:str='IC_activated',
                    cur_activated_key:str='IC_current_activated',
                    prefix:str = "",
                    save_dir = None,
                    show:bool=True):
        activated = g.ndata[activated_key].clone().cpu().detach().numpy().astype(np.int) if activated_key is not None\
             else torch.zeros(g.number_of_nodes(),g.number_of_nodes())
        
        cur_activated = g.ndata[cur_activated_key].clone().cpu().detach().numpy().astype(np.int) if cur_activated_key is not None\
             else torch.zeros(g.number_of_nodes(),g.number_of_nodes())


        activated_map = activated + cur_activated

        fig = plt.figure(figsize=(10,8))
        ax=fig.add_subplot(1,1,1)   
        sns.heatmap(activated_map,ax=ax)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            fig.savefig(os.path.join(save_dir,prefix+'.jpg'))
        if show:
            fig.show()
        
    @staticmethod
    def run(self, g:dgl.DGLHeteroGraph)->dgl.DGLHeteroGraph:
        g = dgl.remove_self_loop(g)
        g.ndata['IC_activated'] = torch.eye(g.number_of_nodes()).bool()
        #g.edata['IC_probability'] = torch.full((g.number_of_edges(),),0.8)
        return IC.activated_graph(
            IC.independent_cascade(g,
                                activated_key='IC_activated',
                                probability_key='IC_probability',
                                sample_times = 5,
                                default_probability=0.5
                                ),
            activated_key='IC_activated'
        )

    def __call__(self,g:dgl.DGLHeteroGraph)->dgl.DGLHeteroGraph:
        return IC.run(g)





if __name__ == '__main__':
    train()
        

    