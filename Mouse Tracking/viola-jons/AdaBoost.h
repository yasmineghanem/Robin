#ifndef ADA_BOOST_H
#define ADA_BOOST_H
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
using namespace std;

class Learner;
class AdaBoost
{
public:
    vector<vector<int>> X;
    vector<int> y;
    vector<double> weights;
    vector<Learner *> learners;
    vector<double> alphas;
    AdaBoost(vector<vector<int>> X, vector<int> y);
    AdaBoost();
    ~AdaBoost();

    void train(int T);
    int predict(const std::vector<int> &X);
    void save(const string file);
    void load(const string file);
    void saveAsText(const string &file);
    void loadFromText(const string &file);
};
#endif