from abc import abstractmethod
import torch
import dgl
import dgl.nn as gnn
import dgl.function as gfn
import numpy as np
import os
import sys

import diffusion.Initializer as Init
import diffusion.ic_api as api
"""
    GraphProcessor is the heart of the project, 
    it set rules for how the diffusion happens.
    To customize the processor, inherit the graphProcessor parent class
    and write the process function
"""


class GraphProcessor:
    '''
        Template of the graph processor
        renew the edges of the origin graph

        retain all the ndata
        renew all the edata
    '''
    def __init__(self):
        pass

    def new_graph(self,
                  g: dgl.DGLHeteroGraph,
                  ind:tuple,
                  ndata:dict = {},
                  edata: dict = {},
                  topo_change: bool = True) -> dgl.DGLHeteroGraph:
        '''
            Build a new graph from the new_key(torch.BoolTensor([num_nodes, num_nodes])) as adj matrix
            Args:
                g,          dgl.DGLHeteroGraph  input graph, 
                ind,        tuple (u,v)   
                edata,      dict, the edata of the new graph
                topo_change, bool, whether to retain the old topology or build the new one
            Returns:
                dgl.DGLHeteroGraph
        '''

        if topo_change:
            new_g = dgl.graph(ind, num_nodes=g.number_of_nodes()).to(g.device)
        else:
            ind = set(i for i in torch.stack(ind,1))  # [num_edge,2]
            old_ind = set(i for i in torch.stack(g.edges(),1))
            ind = old_ind & ind
            ind = torch.stack(list(ind),1)  # [2, num_edge]
            new_g = dgl.graph((ind[0], ind[1]), num_nodes=g.number_of_nodes()).to(g.device)
            
        # copy data
        new_g.__name__ = g.__name__

        if len(ndata) > 0:
            for k, v in ndata.items():
                new_g.ndata[k] = v

        if len(edata) > 0:
            for k, v in edata.items():
                new_g.edata[k] = v

        return new_g

    def aggregate_ndata(self, g: dgl.DGLHeteroGraph, key: str = 'feat'):
        """
            aggregate the g.ndata[key] for a epoch with mean value
        """
        g_ = dgl.add_self_loop(g)
        dim = g.ndata[key].size(-1)
        conv = gnn.GraphConv(dim,dim,weight=False, bias=False, activation=None)
        g.ndata[key] = conv(g_,g.ndata[key])
        return g

    @abstractmethod
    def __call__(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        '''
            rebuild the graph (renew the adj matrix) from the same nodes
        '''
        pass

class IC(GraphProcessor):
    def __init__(self,
                 activated_key: str = "IC_activated",
                 current_activated_key: str = 'IC_current_activated',
                 probability_key: str = "IC_probability",
                 probability_init:Init.EdgeInitializer = Init.EdgeConstant(0.5),
                 batch_size:int = -1,
                 sample_times=2,
                 aggregate: bool = False,
                 verbose:bool = True):
        super(IC, self).__init__()
        self.act_key = activated_key
        self.cur_act_key = current_activated_key
        self.prob_key = probability_key
        self.prob_init = probability_init
        self.batch_size = batch_size
        self.samp_times = sample_times
        self.agg = aggregate
        self.verbose=verbose
        self.__name__ = "IC"

    def independent_cascade(
        self,
        g: dgl.DGLHeteroGraph,
        stop_situation: str = 'dunbar',
        ) -> dgl.DGLHeteroGraph:
        '''
            Args:
                g, dgl.DGLHeteroGraph   
                    g.ndata[activated_key] torch.boolTensor([i])
                    g.edata[probability_key] torch.FloatTensor([1 or i])
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

        def get_parallel_num(th: torch.Tensor) -> int:
            if th.dim() == 1:
                return 1
            else:
                assert th.dim() == 2
                return th.size(1)

        assert stop_situation in ('dunbar', 'iteration')
        assert self.act_key in g.ndata.keys()
        assert self.cur_act_key not in g.ndata.keys()
        assert self.prob_key in g.edata.keys()
        assert (g.edata[self.prob_key] >= 0).all()
        assert self.samp_times > 0

        # align types
        g.ndata[self.act_key] = g.ndata[self.act_key].bool()
        g.edata[self.prob_key] = g.edata[self.prob_key].float()
        g.ndata[self.cur_act_key] = g.ndata[
            self.act_key].clone()  # initialize cur_act_key [num_nodes, num_parallel]

        # broadcast
        parallel_num = get_parallel_num(g.ndata[self.act_key])
        if g.edata[self.prob_key].dim() == 1 and parallel_num > 1:
            old_prob_edata = g.edata[self.prob_key]
            g.edata[self.prob_key] = g.edata[self.prob_key][:, None].repeat(
                1, parallel_num)
        else:
            assert parallel_num == get_parallel_num(g.edata[self.prob_key])

        # Message Passing

        def IC_message(edges):

            return {
                'm':
                edges.src[self.cur_act_key].float() * edges.data[self.prob_key]
            }

        def IC_reduce(nodes):
            # nodes.mailbox['m']  [num_batch, num_neighbor, num_parallel]
            # stop_mask           [num_parallel]
            updated_activated = torch.full_like(
                nodes.mailbox['m'][:, 0, :], False,
                dtype=torch.bool)  # [num_batch,num_parallel]
            for _ in range(self.samp_times):
                sample = (torch.rand_like(nodes.mailbox['m']) <nodes.mailbox['m']).any(1)  # [num_batch, num_parallel]
                updated_activated = updated_activated | sample
            cur_activated = updated_activated & (~nodes.data[self.act_key])
            activated = updated_activated | nodes.data[self.act_key]
            return {
                self.act_key: activated,
                self.cur_act_key: cur_activated
                }

        if stop_situation == 'dunbar':
            iteration = 0
            while not (~g.ndata[self.cur_act_key]).all():
                iteration += 1
                g.update_all(IC_message, IC_reduce)
                g.ndata[self.cur_act_key].T[torch.where(
                    g.ndata[self.act_key].int().sum(0) >= DUNBAR_NUMBER
                )] = False
                print(f"iteration:{iteration}")
        elif stop_situation == 'iteration':
            for _ in range(NUM_ITER):
                g.update_all(IC_message, IC_reduce)

        g.ndata.pop(self.cur_act_key)
        g.edata[self.prob_key] = old_prob_edata
        return g

    def __call__(
        self,
        g: dgl.DGLHeteroGraph,
        ) -> dgl.DGLHeteroGraph:
        '''
            requires feat in g.ndata
            do not change the key or the graph data during this process
        '''

        # init graph
        g = dgl.remove_self_loop(g)  # message from one node cannot be passed to itself
        assert self.act_key not in g.ndata
        assert self.prob_key not in g.edata
        # set the batch size of nodes for duffusion at one time
        self.batch_size = g.number_of_nodes() if self.batch_size == -1 else self.batch_size
        assert self.batch_size > 0
        act_init = Init.ParitialEye() 

        g.edata[self.prob_key] = self.prob_init(g).to(g.device)

        ind = (torch.LongTensor([]),torch.LongTensor([]))
        # 
        for i,base in enumerate(np.arange(0,g.number_of_nodes(),self.batch_size)):

            batch_size = g.number_of_nodes() - base if (base + self.batch_size)>g.number_of_nodes() else self.batch_size

            if self.verbose:
                print(f"batch_iter:{i}, base:{base}, end:{base+batch_size}")

            g.ndata[self.act_key] = act_init(base, batch_size, g).to(g.device)

            # process on graph
            g = self.independent_cascade(g)
            append_ind = torch.where(g.ndata[self.act_key])  # ind in [num_nodes, batch_size]
            ind = torch.cat([ind[0], append_ind[0].cpu()]), torch.cat([ind[1],(append_ind[1] + base).cpu()])
            print(ind)

        g = self.new_graph(g, ind, 
                ndata={
                    "feat":g.ndata['feat'],
                    "label":g.ndata['label'],
                    "train_mask":g.ndata['train_mask'],
                    "val_mask":g.ndata['val_mask'],
                    "test_mask":g.ndata['test_mask']
                }
        )
        # add self loop
        if self.agg:
            g = self.aggregate_ndata(g, key='feat')

        return g


class IC_cpp(GraphProcessor):
    
    def __init__(self,
                 probability_init:Init.EdgeInitializer = Init.EdgeConstant(0.5),
                 topo_change:bool=True,
                 sample_times=2,
                 aggregate: bool = False,
                 verbose:bool = True):
        super(IC_cpp, self).__init__()
        '''
        currently only support cpu
        '''

        
        # load cpp_module

        self.ic = api.ic_slow_cpu

        self.prob_init = probability_init
        self.topo_change = topo_change
        self.samp_times = sample_times
        self.agg = aggregate
        self.verbose=verbose
        self.__name__ = "IC_cpp"

    def __call__(self, g:dgl.DGLHeteroGraph)->dgl.DGLHeteroGraph:
        device = g.device
        ndata = g.ndata
        g = g.cpu()
        # init graph
        g = dgl.remove_self_loop(g)  # message from one node cannot be passed to itself

    

        # diffuse
        g = api.ic_fast_cpu(
                g,
                self.prob_init(g)
                )
    
        # print(new_edge)
        # copy ndata
        g = g.to(device)
        for k,v in ndata.items():
            if k == "feat" or k=='label' or k.endswith('mask'):
                g.ndata[k] = v

        # add self loop
        if self.agg:
            g = self.aggregate_ndata(g, key='feat')

        return g




# class HawkesIC(IC):
    #     def __init__(self,
    #                  hawkes_decay: float = 0.8,
    #                  activated_key: str = "IC_activated",
    #                  hawkes_key: str = "IC_hawkes",
    #                  current_activated_key: str = 'IC_current_activated',
    #                  probability_key: str = "IC_probability",
    #                  activated_init=Init.Eye(),
    #                  probability_init=Init.EdgeConstant(0.5),
    #                  aggregate: bool = False):
    #         super().__init__(activated_key, current_activated_key, probability_key,
    #                          activated_init, probability_init, diffuse)

    #         self.hawkes_key = hawkes_key
    #         self.hawkes_decay = hawkes_decay

    #     def independent_cascade(
    #         self,
    #         g: dgl.DGLHeteroGraph,
    #         stop_situation: str = 'dunbar',
    #         sample_times: int = 1,
    #         ) -> dgl.DGLHeteroGraph:
    #         '''
    #             Args:
    #                 g, dgl.DGLHeteroGraph   
    #                     g.ndata[activated_key] torch.boolTensor([i])
    #                     g.edata[probability_key] torch.FloatTensor([1 or i])
    #                 stop_situation,str  in ['dunbar', 'iteration']  , default: 'dunbar'
    #                 sample_times, int, default: 1

                    
    #             Returns:
    #                 g, dgl.DGLHeteroGraph
    #                     g.ndata[activated_key]
    #                     g.edata[probability_key]
    #         '''
    #         DUNBAR_NUMBER = 16
    #         NUM_ITER = 4

    #         # Assertion

    #         def get_parallel_num(th: torch.Tensor) -> int:
    #             if th.dim() == 1:
    #                 return 1
    #             else:
    #                 assert th.dim() == 2
    #                 return th.size(1)

    #         assert stop_situation in ('dunbar', 'iteration')
    #         assert self.act_key in g.ndata.keys()
    #         assert self.cur_act_key not in g.ndata.keys()
    #         assert self.prob_key in g.edata.keys()
    #         assert (g.edata[self.prob_key] >= 0).all()
    #         assert sample_times > 0

    #         # align types
    #         g.ndata[self.act_key] = g.ndata[self.act_key].float()
    #         g.edata[self.prob_key] = g.edata[self.prob_key].float()

    #         # broadcast
    #         parallel_num = get_parallel_num(g.ndata[self.act_key])
    #         if g.edata[self.prob_key].dim() == 1 and parallel_num > 1:
    #             g.edata[self.prob_key] = g.edata[self.prob_key][:, None].repeat(
    #                 1, parallel_num)
    #         else:
    #             assert parallel_num == get_parallel_num(g.edata[self.prob_key])

    #         # add tmp attributes
    #         g.ndata[self.cur_act_key] = g.ndata[
    #             self.act_key].clone().bool()  # bool [num_nodes, num_parallel]

    #         decay = 1.

    #         # Message Passing

    #         def HawkesIC_message(edges):
    #             expect = edges.src[self.cur_act_key].float() * edges.data[
    #                 self.prob_key]  # [num_edges, num_parallel]
    #             mail = torch.full_like(
    #                 expect, False, dtype=torch.bool)  # [num_edges, num_parallel]
    #             for _ in range(sample_times):
    #                 sample = (torch.rand_like(expect) < expect)
    #                 mail |= sample
    #             return {'m': mail}

    #         def HawkesIC_reduce(nodes):
    #             # nodes.mailbox['m']  [num_batch, num_neighbor, num_parallel]
    #             # stop_mask           [num_parallel]

    #             last_activated = nodes.data[
    #                 self.act_key].bool()  # bool[num_batch, num_parallel]
    #             updated_activated = nodes.mailbox['m'].any(
    #                 1)  # bool [num_batch, num_parallel]
    #             # current activated is in the update list but not activated before.
    #             cur_activated = updated_activated & (
    #                 ~last_activated)  # bool[num_batch, num_parallel]
    #             # activated include the ones activated before and the updated ones.
    #             hawkes_activated = cur_activated.float() * decay + nodes.data[
    #                 self.act_key]

    #             decay *= self.hawkes_decay
    #             return {
    #                 self.act_key: hawkes_activated,
    #                 self.cur_act_key: cur_activated
    #             }

    #         if stop_situation == 'dunbar':
    #             iteration = 0
    #             while not (~g.ndata[self.cur_act_key]).all():
    #                 iteration += 1
    #                 g.update_all(HawkesIC_message, HawkesIC_reduce)
    #                 # mask current activated node by measuring the activated node with dunbar_number
    #                 g.ndata[self.cur_act_key].T[torch.where(
    #                     g.ndata[self.act_key].bool().int().sum(0) >= DUNBAR_NUMBER
    #                 )] = False
    #                 print(f"iteration:{iteration}")
    #         elif stop_situation == 'iteration':
    #             for _ in range(NUM_ITER):
    #                 g.update_all(HawkesIC_message, HawkesIC_reduce)

    #         g.ndata.pop(self.cur_act_key)
    #         return g


    #     def __call__(
    #         self,
    #         g: dgl.DGLHeteroGraph,
    #         ) -> dgl.DGLHeteroGraph:
    #         # initialize graph
    #         # initialize graph
    #         g = dgl.remove_self_loop(g)
    #         assert self.act_key not in g.ndata
    #         assert self.cur_act_key not in g.ndata
    #         assert self.prob_key not in g.edata

    #         self.act_init(g, self.act_key)
    #         self.prob_init(g, self.prob_key)

    #         # process on graph
    #         g = self.independent_cascade(g)
    #         g = self.new_graph(g,
    #                            new_key=self.act_key,
    #                            edata={
    #                                self.hawkes_key:
    #                                g.ndata[self.act_key][torch.where(
    #                                    g.ndata[self.act_key].bool())]
    #                            })

                            
    #         if self.agg:
    #             g = self.aggregate_ndata(g, key='feat')

