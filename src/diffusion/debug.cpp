# include "debug.h"


template<typename T>
void print_brief(std::vector<std::vector<T>> adj){

# if(DUMP)    
std::ofstream log_file("debug_print_brief.log",std::ios::out);
# endif    
    for(int i=0;i<adj.size();++i){
        std::cout<<adj[i].size()<<",";
# if(DUMP)
        log_file<<adj[i].size()<<",";
#endif
    }
    std::cout<<std::endl;
}


template<typename T>
void log_2d(std::vector<std::vector<T>> prob){
# if(DUMP)
    std::ofstream log("debug_print_detail.log",std::ios::out);
# endif
    for(int i=0;i<prob.size();++i){
        for(int j=0;j<prob[i].size();++j){
            std::cout<<prob[i][j]<<",";
# if(DUMP)
            log<<prob[i][j]<<",";
# endif
        }
        std::cout<<"||";
# if(DUMP)
        log<<"||";
# endif
    }
    std::cout<<std::endl;
}
