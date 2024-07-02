#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
using namespace std;

class Learner;

void integral_image(const vector<vector<int>> &I, vector<vector<int>> &II);
int sum_region(const vector<vector<int>> &ii, int x1, int y1, int x2, int y2);
vector<int> compute_haar_like_features(const vector<vector<int>> &img, const vector<vector<int>> &II);
int haar_feature_scaling(const vector<vector<int>> &image, const string &feature_type, int i, int j, int w, int h);
Learner *decision_stump(vector<vector<int>> &X, const vector<int> &y, const vector<double> &weights, int feature_index, vector<int> &sorted_indices, vector<vector<int> *> &X_sorted, vector<int> &y_sorted, vector<double> &weights_sorted, vector<double> &pos_weights_prefix, vector<double> &neg_weights_prefix);
Learner *best_stump(vector<vector<int>> &X, const vector<int> &y, const vector<double> &weights, int num_features);
std::vector<std::string> get_files(const std::string &path, int num = -1);
void load_gray_images(const std::string &path, vector<vector<vector<int>>> &images, int num);

#endif