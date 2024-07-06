#ifndef ADA_BOOST_H
#define ADA_BOOST_H
// #include <vector>
#include <cmath>
#include <string>
#include <algorithm>
using namespace std;

class Learner;
class AdaBoost
{
public:
    int **X;
    int *y;
    double *weights;
    pair<int, int> train_dim;
    vector<Learner *> learners;
    vector<double> alphas;
    AdaBoost(int **&X, int *&y, pair<int, int> train_dim);
    AdaBoost();
    ~AdaBoost();

    void train(int T);
    int predict(int *&X, double sl = 0);
    int predict(int **&X, int size, double sl = 0);
    void save(const string file);
    void load(const string file);
    void saveAsText(const string &file);
    void loadFromText(const string &file);
};
#endif