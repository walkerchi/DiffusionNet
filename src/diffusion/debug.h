#include <ctime>
#include <iostream>

class Timer{
    private:
        clock_t start;
    public:
    void begin(){
        start = clock();
    }

    double end(){
        clock_t finish = clock();
        double duration;
        duration = (double)(finish-start)/CLOCKS_PER_SEC;
        std::cout<<"consumes:"<<duration<<"s"<<std::endl;
    }
};