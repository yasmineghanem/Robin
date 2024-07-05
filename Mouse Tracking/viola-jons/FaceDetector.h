#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H
// #include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "AdaBoost.h"
#include "learner.h"
#include <tuple>
using namespace std;

class FaceDetector
{
private:
    vector<AdaBoost> cascade;
    vector<double> shif;

    int **X_train;
    int *y_train;
    pair<int, int> train_dim;
    int ***X_val;
    int *y_val;
    tuple<int, int, int> val_dim;

public:
    FaceDetector(int **&X_train, int *&y_train, int ***&X_val, int *&y_val, pair<int, int> train_dim, tuple<int, int, int> val_dim);
    FaceDetector();
    // yo : desired overall false positive rate
    // yl : desired targeted layer false positive
    // Bl : desired targeted layer false negative , 1-Bl detection rate
    void train(double Yo, double Yl, double Bl);
    matrices evaluate_single_layer(AdaBoost &fl, int *&predictions, double sl);
    void remove_negative_train_data();
    void remove_negative_val_data();
    int predict(int **&img);
    void save(const string folder);
    void load(const string folder);
};

#endif
