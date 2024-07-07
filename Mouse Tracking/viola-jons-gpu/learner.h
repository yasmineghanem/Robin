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

    __host__ __device__ Learner(double threshold, int polarity, double error, double margin, int feature_index);
    __host__ __device__ Learner();
    __host__ __device__ int predict(int *&X);
    __host__ __device__ int predict(int **&X, int size, double devide = 1.0);
};

#endif