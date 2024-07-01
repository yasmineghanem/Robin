#include "learner.h"

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
int Learner::predict(const vector<double> &X)
{
    return this->polarity * ((X[this->feature_index] - this->threshold) >= 0 ? 1 : -1);
}