#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H
#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "AdaBoost.h"
#include "learner.h"
using namespace std;

class FaceDetector
{
private:
    vector<AdaBoost> cascade;
    vector<double> shif;
    int n_train;
    int n_validation;

    vector<vector<int>> X_train;
    vector<int> y_train;
    vector<vector<vector<int>>> X_val;
    vector<int> y_val;

public:
    FaceDetector(vector<vector<int>> &X_train, vector<int> &y_train, vector<vector<vector<int>>> &X_val, vector<int> &y_val);
    FaceDetector();
    // yo : desired overall false positive rate
    // yl : desired targeted layer false positive
    // Bl : desired targeted layer false negative , 1-Bl detection rate
    void train(double Yo, double Yl, double Bl);
    matrices evaluate_single_layer(AdaBoost &fl, vector<int> &predictions, double sl);
    void remove_false_data();
    int predict(vector<vector<int>> &img);
    void save(const string folder);
    void load(const string folder);
};

#endif
