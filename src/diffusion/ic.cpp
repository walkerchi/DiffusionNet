
/*
 Pytorch Cpp Extension of independent_cascade
*/
#include <torch/extension.h>
#include <iostream>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include "debug.h"

namespace py = pybind11;

namespace IC{
    std::array<at::Tensor,2> ic_slow_cpu(
        int num_node,
        at::Tensor src,
        at::Tensor dst,
        at::Tensor prob,
        bool topo_change,
        int sample_times,  // sample times must >= 1
        int dunbar   // dunbar must be positive
    ){
        /*
         initailize
        */
        // input
        Timer t;
        t.begin();

        int i,j;
        int cur_node,  cur_edge;
        int num_edge = src.numel();
        int new_num_edge = 0;
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<> distr(0,1);
        std::vector<std::vector<int>> adj(num_node, std::vector<int>());
        std::vector<std::vector<float>> p(num_node, std::vector<float>());
        // mid
        std::vector<std::vector<int>> activated(num_node, std::vector<int>());

        std::cout<<"init ";
        t.end();

        

        /*
        implement adj and p 
        it's like transfer the scipy.sparse.coo to scipy.sparse.csr
        coo 2 csr   also edge to adj and prob to p
        adj[cur_node] = neighbor ids of cur_node
        p[cur_node] = probability from cur_node to neighbor
        */
        t.begin();
        std::cout<<"num_edge:"<<num_edge<<std::endl;
        std::cout<<"dunbar:"<<dunbar<<std::endl;
        

        for(cur_edge=0 ; cur_edge < num_edge ; ++cur_edge){
            adj[src[cur_edge].item<int>()].push_back(dst[cur_edge].item<int>());
            p[src[cur_edge].item<int>()].push_back(prob[cur_edge].item<float>());
        }    

        std::cout<<"transfer ";
        t.end();
        // print_brief(adj);
        // print_detail(adj);
        // print_prob(p);


        /*
        diffusion similar to BFS
        */
        t.begin();

        for(cur_node=0; cur_node<num_node ;++cur_node){
            // for each node
            std::vector<int> act{};    // store activated ones
            std::queue<int> cur_act{}; // store currently active ones
            cur_act.push(cur_node);
            

            while(!cur_act.empty() && act.size()<dunbar){

                // pop u from cur_act 
                // each edge could be represented as (u,v)
                // std::cout<<cur_act.size()<<"  "<<cur_act.front()<<std::endl;
                int u = cur_act.front(); 
                cur_act.pop();

                
                for(int v_ind=0; 
                    v_ind<adj[u].size() && act.size()<dunbar; 
                    ++v_ind){
                    // for each neighbor of u
                    int v = adj[u][v_ind];
                    
                    // assure v not in act;
                    int v_in_act = 0;
                    for(i=0; i<act.size(); ++i){
                        if(act[i] == v){
                            v_in_act = 1;
                            break;
                        }
                    }
                    if(v_in_act)
                        break;
                    
                    // sample
                    for(i=0; 
                        i<sample_times && distr(eng)<p[u][v_ind]; 
                        ++i){
                        cur_act.push(v);
                        act.push_back(v);
                    }


                    // neighbor loop end
                }

                // queue loop end
            }
            
            // node loop end
            activated[cur_node] = act;
            new_num_edge  += act.size();
        }

        std::cout<<"diffusion ";
        t.end();



        /*
        csr 2 coo
        */
        t.begin();
        std::cout<<"new_num_edge:"<<new_num_edge<<std::endl;

        // log_2d(activated);
        // print_brief(activated);
        // for(i=0;i<activated.size();++i){
        //     for(j=0;j<activated[i].size();++j){
        //         std::cout<<activated[i][j]<<",";
        //     }
        //     std::cout<<"||";
        // }

        at::Tensor new_src = torch::zeros({new_num_edge}, torch::kInt32);
        at::Tensor new_dst = torch::zeros({new_num_edge}, torch::kInt32);
        int count = 0;
        for(i=0; i<num_node; ++i){
            for(j=0; j<activated[i].size(); ++j, ++count){
                new_src[count] = i;
                new_dst[count] = activated[i][j];
            }
        }

        std::cout<<"transfer ";
        t.end();

        std::array<at::Tensor,2> new_edge = {new_src, new_dst};
        return new_edge;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("ic_slow_cpu",&IC::ic_slow_cpu,py::return_value_policy::reference);
}