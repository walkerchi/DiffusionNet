# include <vector>
# include <iostream>
# include <ctime>
# include <fstream>

# define DUMP 0

template<typename T>
void print_brief(std::vector<std::vector<T>> adj);


template<typename T>
void log_2d(std::vector<std::vector<T>> prob);

class Timer{

    clock_t st_time;
    public:
        void begin(){ // start clocking
            this->st_time = clock();
        }
        float end(bool verbose=true){ // end clocking
            clock_t finish = clock();
            float duration = (double)(finish - this->st_time) / CLOCKS_PER_SEC;
            if(verbose) 
                std::cout<<"consums:"<<duration<<"s"<<std::endl;
            return duration;
        }
};