#include "learner.h"
#include <iostream>
using namespace std;
#include "const.h"
#include "utils.h"
Learner::Learner(double threshold, int polarity, double error, double margin, int feature_index)
{
    this->threshold = threshold;
    this->polarity = polarity;
    this->error = error;
    this->margin = margin;
    this->feature_index = feature_index;
}
Learner::Learner()
{
    this->threshold = 0;
    this->polarity = 1;
    this->error = 2;
    this->margin = 0;
    this->feature_index = 0;
}
int Learner::predict(int *&X)
{
    return this->polarity * ((X[this->feature_index] >= this->threshold) ? 1 : -1);
}
int Learner::predict(int **&X, int size)
{
    int index = this->feature_index;
    return this->polarity * (haar_feature_scaling(X, size, features_info[index].feature_type, features_info[index].i, features_info[index].j, features_info[index].w, features_info[index].h) >= this->threshold ? 1 : -1);
    return 0;
}
