#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
using namespace std;

class Learner;

void integral_image(const vector<vector<double>> &I, vector<vector<double>> &II);
int sum_region(const vector<vector<double>> &ii, int x1, int y1, int x2, int y2);
vector<double> compute_haar_like_features(const vector<vector<double>> &img, const vector<vector<double>> &II);
double haar_feature_scaling(const vector<vector<double>> &image, const string &feature_type, int i, int j, int w, int h);
Learner *decision_stump(vector<vector<double>> &X, const vector<int> &y, const vector<double> &weights, int feature_index, vector<int> &sorted_indices, vector<vector<double> *> &X_sorted, vector<int> &y_sorted, vector<double> &weights_sorted);
Learner *best_stump(vector<vector<double>> &X, const vector<int> &y, const vector<double> &weights, int num_features);

#endif