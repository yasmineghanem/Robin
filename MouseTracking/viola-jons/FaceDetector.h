#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H
#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "AdaBoost.h"
#include "learner.h"
#include <tuple>
#include "const.h"
#include <cmath>
using namespace std;

class FaceDetector
{
private:
    vector<AdaBoost *> cascade;
    vector<double> shif;

    int **X_train;
    int *y_train;
    pair<int, int> train_dim;
    int ***X_val;
    int *y_val;
    tuple<int, int, int> val_dim;
    string folder;
    int stride = 1;
    float smallest_box = 1.0 / 1000;
    int min_confidence = 3;

public:
    FaceDetector(int **&X_train, int *&y_train, int ***&X_val, int *&y_val, pair<int, int> train_dim, tuple<int, int, int> val_dim, string save_folder);
    FaceDetector();
    ~FaceDetector();
    // yo : desired overall false positive rate
    // yl : desired targeted layer false positive
    // Bl : desired targeted layer false negative , 1-Bl detection rate
    void train(double Yo, double Yl, double Bl);
    matrices evaluate_single_layer(AdaBoost *fl, int *&predictions, double sl);
    void remove_negative_train_data();
    void remove_negative_val_data();
    int predict(int **&img, int size, double devide = 1.0);
    void save(const string folder);
    void load(const string folder);
    // M is the number of rows "height"
    // N is the number of columns "width"
    vector<window *> process(int **&img, int ***&color_img, int M, int N, double c = 1.5);
    // test on the variance of the window, test on the prediction of the model on the window
    bool window_test1(window *win, int **II, long long **IIsq);

    // test on the connected components of the window,and overlaped windows
    // the min number of connected windws in the connected componnet to pretect there is face in this commponent
    void window_test2(vector<window *> &windows, int **&img, int M, int N);
    void window_test3(vector<window *> &windows, int **&img, int M, int N);
};

#endif
