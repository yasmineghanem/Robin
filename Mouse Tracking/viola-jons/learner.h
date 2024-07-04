#ifndef LEANER_H
#define LEANER_H
// #include <vector>
using namespace std;
class Learner
{
public:
    double threshold;
    int polarity;
    double error;
    double margin;
    int feature_index = 0;
    Learner(double threshold, int polarity, double error, double margin, int feature_index);
    Learner();
    int predict(int *&X);
};

#endif