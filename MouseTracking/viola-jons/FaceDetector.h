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
typedef struct result
{
    vector<window *> windows;
} result;

typedef struct input
{
    int **II = 0;
    long long **IIsq = 0;
    int **skin_denisty = 0;
    int **img = 0;
    int ***color_img = 0;
    int M = 0;
    int N = 0;
    double c = 1.1;
    int starti = 0;
    int startj = 0;
    int endi = 0;
    int endj = 0;
    int max_size = 1000;
    int num_threads = 2;
} input;
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
    float smallest_box = 1.0 / 100;
    int min_confidence = 13;
    double cur_Y = 1;

public:
    bool start_flag = false;
    bool end_flag = false;
    result res;
    input in;
    int stride = 2;

    void set_in(int **II = 0,
                long long **IIsq = 0,
                int **skin_denisty = 0,
                int **img = 0,
                int ***color_img = 0,
                int M = 0,
                int N = 0,
                double c = 1.1,
                int starti = 0,
                int startj = 0,
                int endi = 0,
                int endj = 0,
                int max_size = 1000,
                int num_threads = 2)
    {
        in.II = II;
        in.IIsq = IIsq;
        in.skin_denisty = skin_denisty;
        in.img = img;
        in.color_img = color_img;
        in.M = M;
        in.N = N;
        in.c = c;
        in.starti = starti;
        in.startj = startj;
        in.endi = endi;
        in.endj = endj;
        in.max_size = max_size;
        in.num_threads = num_threads;
    }
    // get result
    vector<window *> get_res()
    {
        return res.windows;
    }
    FaceDetector(int **&X_train, int *&y_train, int ***&X_val, int *&y_val, pair<int, int> train_dim, tuple<int, int, int> val_dim, string save_folder);
    FaceDetector();
    ~FaceDetector();
    // yo : desired overall false positive rate
    // yl : desired targeted layer false positive
    // Bl : desired targeted layer false negative , 1-Bl detection rate
    void train(double Yo, double Yl, double Bl, bool restric = false);
    matrices evaluate_single_layer(AdaBoost *fl, int *&predictions, double sl);
    void remove_negative_train_data();
    void remove_negative_val_data();
    int predict(int **&img, int size, double devide = 1.0);
    void save(const string folder);
    void load(const string folder);
    // M is the number of rows "height"
    // N is the number of columns "width"
    vector<window *> process(int **&img, int ***&color_img, int M, int N, double c = 1.1);
    vector<window *> process_part(int **II, long long **IIsq, int **skin_denisty, int **&img, int ***&color_img, int M, int N, double c = 1.1, int starti = 0, int startj = 0, int endi = 0, int endj = 0, int max_size = 1000, int num_threads = 2);
    void infinite_prcess();
    // test on the variance of the window, test on the prediction of the model on the window
    bool window_test1(window *win, int **II, long long **IIsq);
    // test on the connected components of the window,and overlaped windows
    // the min number of connected windws in the connected componnet to pretect there is face in this commponent
    void window_test2(vector<window *> &windows, int **&img, int M, int N);
    void window_test3(vector<window *> &windows, int **&img, int M, int N);
    void rebuild();
};

#endif
