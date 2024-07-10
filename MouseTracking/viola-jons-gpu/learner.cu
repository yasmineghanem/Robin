#include "learner.h"
#include <iostream>
using namespace std;
#include "const.h"
#include "utils.h"
#include <vector>
#include <cuda_runtime.h>
__host__ __device__ int Learner::predict(int *&X)
{
    return this->polarity * ((X[this->feature_index] >= this->threshold) ? 1 : -1);
}
__host__ __device__ int Learner::predict(int **&X, int size, double devide)
{
    int index = this->feature_index;
#ifdef __CUDA_ARCH__
    // On the device, use the device pointer
    return this->polarity * ((haar_feature_scaling(X, size, d_features_info[index].feature_type, d_features_info[index].i, d_features_info[index].j, d_features_info[index].w, d_features_info[index].h) / devide) >= this->threshold ? 1 : -1);
#else
    // On the host, use the host pointer
    return this->polarity * ((haar_feature_scaling(X, size, features_info[index].feature_type, features_info[index].i, features_info[index].j, features_info[index].w, features_info[index].h) / devide) >= this->threshold ? 1 : -1);
#endif
}
