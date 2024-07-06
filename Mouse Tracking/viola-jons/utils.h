#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;

class Learner;
typedef struct matrices
{
    double accuracy;
    double error_rate;
    double false_positive_rate;
    double false_negative_rate;
} matrices;

void integral_image(int **&I, int **&II, int h, int w);
int sum_region(int **&ii, int x1, int y1, int x2, int y2);
int compute_haar_like_features(int **&II, int *&features);
void fill_features_info();
int haar_feature_scaling(int **&image, int size, const char &feature_type, int i, int j, int w, int h);
Learner *decision_stump(int **&X, int *&y, double *&weights, int feature_index, int *sorted_indices, int *X_sorted, int *Y_sorted, double *weights_sorted, double *&pos_weights_prefix, double *&neg_weights_prefix, pair<int, int> &dim);
Learner *best_stump(int **&X, int *&y, double *&weights, pair<int, int> &dim);
Learner *best_stump_threads(int **&X, int *&y, double *&weights, std::pair<int, int> &dim);
int get_files(const std::string &path, string *&files, int num);
int load_gray_images(const std::string &path, int ***&images, int num);
void load_haar_like_features(const string &path, int **&X, int *&Y, int num, int y_label);
pair<int, int> load_features(const string &pos_path, const string &neg_path, int **&X, int *&Y, int pos_num, int neg_num);
matrices calc_acuracy_metrices(int *Y_test, int *predeictions, int count);

#endif