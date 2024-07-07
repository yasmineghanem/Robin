#ifndef LEANER_H
#define LEANER_H
#include <vector>
#include <cuda_runtime.h>
using namespace std;
class Learner
{
public:
    double threshold;
    int polarity;
    double error;
    double margin;
    int feature_index = 0;

    __host__ __device__ Learner::Learner(double threshold, int polarity, double error, double margin, int feature_index)
    {
        this->threshold = threshold;
        this->polarity = polarity;
        this->error = error;
        this->margin = margin;
        this->feature_index = feature_index;
    }
    __host__ __device__ Learner::Learner()
    {
        this->threshold = 0;
        this->polarity = 1;
        this->error = 2;
        this->margin = 0;
        this->feature_index = 0;
    }
    __host__ __device__ int predict(int *&X);
    __host__ __device__ int predict(int **&X, int size, double devide = 1.0);
};

#endif