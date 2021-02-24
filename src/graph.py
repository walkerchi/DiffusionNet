from abc import abstractmethod
import torch
import dgl
import dgl.function as gfn


class GraphProcessor:
    '''
        Template of the graph processor
        renew the edges of the origin graph

        retain all the ndata
        renew all the edata
    '''
    def __init__(self):
        pass 
    
    @staticmethod
    def new_graph(g:dgl.DGLHeteroGraph, new_key:str='activated', edata:dict={}, topo_change:bool=True)->dgl.DGLHeteroGraph:
        '''
            Build a new graph from the activated_key(torch.BoolTensor([num_nodes, num_nodes])) as adj matrix
            Args:
                g,          dgl.DGLHeteroGraph  input graph, 
                renew_key,  str    
                edata,      dict, the edata of the new graph
                topo_change, bool, whether to retain the old topology or build the new one
            Returns:
                dgl.DGLHeteroGraph
        '''
        new_adj = g.ndata[new_key].bool()
        assert new_adj.shape == (
            g.number_of_nodes(), g.number_of_nodes()
            ),f"activated shape: {new_adj.shape}  node num:{g.number_of_nodes()}"
        
        if not topo_change:
            new_adj = new_adj & g.adj().to_dense().bool()

        new_g = dgl.graph(torch.where(new_adj), num_nodes=g.number_of_nodes())

        for k, v in g.ndata.items():
            if k != new_key:      
                new_g.ndata[k] = v
           
        if len(edata) > 0:
            new_g.edata = edata

        return new_g

    @staticmethod
    @abstractmethod
    def process(g:dgl.DGLHeteroGraph)->dgl.DGLHeteroGraph:
        '''
            rebuild the graph (renew the adj matrix) from the same nodes
        '''
        pass




class IC(GraphProcessor):
    def __init__(self):
        super(IC,self).__init__()
        pass 

    @staticmethod
    def independent_cascade(g:dgl.DGLHeteroGraph,
                    activated_key:str='IC_activated',
                    probability_key:str='IC_probability',
                    stop_situation:str='dunbar',
                    sample_times:int = 1,
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




        # append IC_current_activated

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
    def process(g:dgl.DGLHeteroGraph)->dgl.DGLHeteroGraph:
        g = dgl.remove_self_loop(g)
        g.ndata['IC_activated'] = torch.eye(g.number_of_nodes()).bool()
        #g.edata['IC_probability'] = torch.full((g.number_of_edges(),),0.8)
        return IC.new_graph(
            IC.independent_cascade(g,
                                activated_key='IC_activated',
                                probability_key='IC_probability',
                                sample_times = 5,
                                default_probability=0.5
                                ),
            new_key='IC_activated'
        )



