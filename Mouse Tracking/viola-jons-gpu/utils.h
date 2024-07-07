#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include "const.h"
#include "learner.h"
#include <cuda_runtime.h>
using namespace std;

typedef struct matrices
{
    double accuracy;
    double error_rate;
    double false_positive_rate;
    double false_negative_rate;
} matrices;

template <typename T>
void integral_image(T **&I, T **&II, int h, int w)
{
    int N = h;
    int M = w;

    II[0][0] = I[0][0];

    // Compute the first row
    for (int j = 1; j < M; j++)
    {
        II[0][j] = I[0][j] + II[0][j - 1];
    }

    // Compute the first column
    for (int i = 1; i < N; i++)
    {
        II[i][0] = I[i][0] + II[i - 1][0];
    }

    // Compute the rest of the integral image
    for (int i = 1; i < N; i++)
    {
        for (int j = 1; j < M; j++)
        {
            II[i][j] = I[i][j] + II[i][j - 1] + II[i - 1][j] - II[i - 1][j - 1];
        }
    }
}

template <typename T>
T sum_region(T **&ii, int x1, int y1, int x2, int y2)
{
    T A = (x1 > 0 && y1 > 0) ? ii[x1 - 1][y1 - 1] : 0;
    T B = (x1 > 0) ? ii[x1 - 1][y2] : 0;
    T C = (y1 > 0) ? ii[x2][y1 - 1] : 0;
    T D = ii[x2][y2];
    return D - B - C + A;
}

int compute_haar_like_features(int **&II, int *&features);
void fill_features_info();
int haar_feature_scaling(int **&image, int size, const char &feature_type, int i, int j, int w, int h);
Learner *decision_stump(int **&X, int *&y, double *&weights, int feature_index, int *sorted_indices, int *X_sorted, int *Y_sorted, double *weights_sorted, double *&pos_weights_prefix, double *&neg_weights_prefix, pair<int, int> &dim);
__device__ Learner *decision_stump_kernal(int *&X, int *&y, float *&weights, int feature_index, int *sorted_indices, int *X_sorted, int *Y_sorted, float *weights_sorted, float *&pos_weights_prefix, float *&neg_weights_prefix, pair<int, int> &dim);
__global__ void decision_stump_GPU(int *X, int *_y_, float *_weights_, pair<int, int> dim, Learner *d_stumps, float *_pos_weights_prefix_, float *_neg_weights_prefix_, int *_sorted_indices_, int *_X_sorted_, int *_Y_sorted_, float *_weights_sorted_);
Learner *best_stump(int **&X, int *&y, double *&weights, pair<int, int> &dim);
Learner *best_stump_threads(int **&X, int *&y, double *&weights, std::pair<int, int> &dim);
Learner *best_stump_GPU(int **&X, int *&y, float *&weights, pair<int, int> &dim) ;
int get_files(const std::string &path, string *&files, int num);
int load_gray_images(const std::string &path, int ***&images, int num);
void load_haar_like_features(const string &path, int **&X, int *&Y, int num, int y_label);
pair<int, int> load_features(const string &pos_path, const string &neg_path, int **&X, int *&Y, int pos_num, int neg_num);
matrices calc_acuracy_metrices(int *Y_test, int *predeictions, int count);
////////////////////// PROCESSING FUNCTIONS for detect from frame ////////////////////////
void saveImageAsPNG(const char *filename, int ***&color_img, int **&img, int M, int N, bool colored = true);
void load_image(const std::string &path, int ***&color_img, int **&img, int &M, int &N);
void drawGreenRectangles(int ***&color_img, int M, int N, std::vector<window *> &P, int thickness = 3);

#endif