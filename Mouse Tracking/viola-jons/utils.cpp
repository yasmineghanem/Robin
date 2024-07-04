#include "utils.h"
#include <iostream>
// #include <vector>
#include <cmath>
#include <algorithm>
#include "learner.h"
#include <chrono>
#include <string>
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "const.h"
using namespace std;
namespace fs = std::filesystem;

using namespace std;
// using namespace std::chrono;
const int step_size = 4;
const int positive_pretection = 1;
const int negative_pretection = -1;
const double err = 1e-6;
auto start_time = std::chrono::high_resolution_clock::now();
auto end_time = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration;

void integral_image(int **&I, int **&II, int h, int w)
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

int sum_region(int **&ii, int x1, int y1, int x2, int y2)
{
    int A = (x1 > 0 && y1 > 0) ? ii[x1 - 1][y1 - 1] : 0;
    int B = (x1 > 0) ? ii[x1 - 1][y2] : 0;
    int C = (y1 > 0) ? ii[x2][y1 - 1] : 0;
    int D = ii[x2][y2];
    return D - B - C + A;
}

int compute_haar_like_features(int **&II, int *&features)
{
    // assert(img.size() == 24 && img[0].size() == 24);

    features = new int[FEATURE_NUM];
    int f = 0;

    // Feature type (a)
    for (int i = 1; i < 25; i++)
    {
        for (int j = 1; j < 25; j++)
        {
            for (int w = 1; w < (25 - j) / 2 + 1; w++)
            {
                for (int h = 1; h < 25 - i + 1; h++)
                {
                    int S1 = sum_region(II, i - 1, j - 1, i - 1 + h - 1, j - 1 + w - 1);
                    int S2 = sum_region(II, i - 1, j - 1 + w, i - 1 + h - 1, j - 1 + 2 * w - 1);
                    features[f] = (S1 - S2);
                    f++;
                }
            }
        }
    }

    // Feature type (b)
    for (int i = 1; i < 25; i++)
    {
        for (int j = 1; j < 25; j++)
        {
            for (int w = 1; w < (25 - j) / 3 + 1; w++)
            {
                for (int h = 1; h < 25 - i + 1; h++)
                {
                    int S1 = sum_region(II, i - 1, j - 1, i - 1 + h - 1, j - 1 + w - 1);
                    int S2 = sum_region(II, i - 1, j - 1 + w, i - 1 + h - 1, j - 1 + 2 * w - 1);
                    int S3 = sum_region(II, i - 1, j - 1 + 2 * w, i - 1 + h - 1, j - 1 + 3 * w - 1);
                    features[f] = (S1 - S2 + S3);
                    f++;
                }
            }
        }
    }

    // Feature type (c)
    for (int i = 1; i < 25; i++)
    {
        for (int j = 1; j < 25; j++)
        {
            for (int w = 1; w < 25 - j + 1; w++)
            {
                for (int h = 1; h < (25 - i) / 2 + 1; h++)
                {
                    int S1 = sum_region(II, i - 1, j - 1, i - 1 + h - 1, j - 1 + w - 1);
                    int S2 = sum_region(II, i - 1 + h, j - 1, i - 1 + 2 * h - 1, j - 1 + w - 1);
                    features[f] = (S1 - S2);
                    f++;
                }
            }
        }
    }

    // Feature type (d)
    for (int i = 1; i < 25; i++)
    {
        for (int j = 1; j < 25; j++)
        {
            for (int w = 1; w < 25 - j + 1; w++)
            {
                for (int h = 1; h < (25 - i) / 3 + 1; h++)
                {
                    int S1 = sum_region(II, i - 1, j - 1, i - 1 + h - 1, j - 1 + w - 1);
                    int S2 = sum_region(II, i - 1 + h, j - 1, i - 1 + 2 * h - 1, j - 1 + w - 1);
                    int S3 = sum_region(II, i - 1 + 2 * h, j - 1, i - 1 + 3 * h - 1, j - 1 + w - 1);
                    features[f] = (S1 - S2 + S3);
                    f++;
                }
            }
        }
    }

    // Feature type (e)
    for (int i = 1; i < 25; i++)
    {
        for (int j = 1; j < 25; j++)
        {
            for (int w = 1; w < (25 - j) / 2 + 1; w++)
            {
                for (int h = 1; h < (25 - i) / 2 + 1; h++)
                {
                    int S1 = sum_region(II, i - 1, j - 1, i - 1 + h - 1, j - 1 + w - 1);
                    int S2 = sum_region(II, i - 1 + h, j - 1, i - 1 + 2 * h - 1, j - 1 + w - 1);
                    int S3 = sum_region(II, i - 1, j - 1 + w, i - 1 + h - 1, j - 1 + 2 * w - 1);
                    int S4 = sum_region(II, i - 1 + h, j - 1 + w, i - 1 + 2 * h - 1, j - 1 + 2 * w - 1);
                    features[f] = (S1 - S2 - S3 + S4);
                    f++;
                }
            }
        }
    }

    return f;
}

int haar_feature_scaling(const vector<vector<int>> &image, const string &feature_type, int i, int j, int w, int h)
{
    int e = image.size();
    // assert(e >= 24);

    auto round_nearest_integer = [](double z)
    {
        return int(round(z));
    };

    if (feature_type == "a")
    {
        double a = 2 * w * h;
        i = round_nearest_integer(i * e / 24);
        j = round_nearest_integer(j * e / 24);
        h = round_nearest_integer(h * e / 24);
        w = 0;
        for (int k = 1; k < round_nearest_integer(1 + 2 * w * e / 24) / 2 + 1; k++)
        {
            if (2 * k <= e - j + 1)
            {
                w = k;
            }
        }
        double S1 = 0;
        double S2 = 0;
        for (int x = i; x < i + h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S1 += image[x][y];
            }
            for (int y = j + w; y < j + 2 * w; y++)
            {
                S2 += image[x][y];
            }
        }
        return (S1 - S2) * a / (2 * w * h);
    }
    else if (feature_type == "b")
    {
        double a = 3 * w * h;
        i = round_nearest_integer(i * e / 24);
        j = round_nearest_integer(j * e / 24);
        h = round_nearest_integer(h * e / 24);
        w = 0;
        for (int k = 1; k < round_nearest_integer(1 + 3 * w * e / 24) / 3 + 1; k++)
        {
            if (3 * k <= e - j + 1)
            {
                w = k;
            }
        }
        double S1 = 0;
        double S2 = 0;
        double S3 = 0;
        for (int x = i; x < i + h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S1 += image[x][y];
            }
            for (int y = j + w; y < j + 2 * w; y++)
            {
                S2 += image[x][y];
            }
            for (int y = j + 2 * w; y < j + 3 * w; y++)
            {
                S3 += image[x][y];
            }
        }
        return (S1 - S2 + S3) * a / (3 * w * h);
    }
    else if (feature_type == "c")
    {
        double a = 2 * w * h;
        i = round_nearest_integer(i * e / 24);
        j = round_nearest_integer(j * e / 24);
        w = round_nearest_integer(w * e / 24);
        h = 0;
        for (int k = 1; k < round_nearest_integer(1 + 2 * h * e / 24) / 2 + 1; k++)
        {
            if (2 * k <= e - i + 1)
            {
                h = k;
            }
        }
        double S1 = 0;
        double S2 = 0;
        for (int x = i; x < i + h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S1 += image[x][y];
            }
        }
        for (int x = i + h; x < i + 2 * h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S2 += image[x][y];
            }
        }
        return (S1 - S2) * a / (2 * w * h);
    }
    else if (feature_type == "d")
    {
        double a = 3 * w * h;
        i = round_nearest_integer(i * e / 24);
        j = round_nearest_integer(j * e / 24);
        w = round_nearest_integer(w * e / 24);
        // h = max(k for k in range(1, round_nearest_integer(1 + 3 * h * e / 24) // 3 + 1) if 3 * k <= e - i + 1)
        h = 0;
        for (int k = 1; k < round_nearest_integer(1 + 3 * h * e / 24) / 3 + 1; k++)
        {
            if (3 * k <= e - i + 1)
            {
                h = k;
            }
        }
        double S1 = 0;
        double S2 = 0;
        double S3 = 0;
        for (int x = i; x < i + h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S1 += image[x][y];
            }
        }
        for (int x = i + h; x < i + 2 * h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S2 += image[x][y];
            }
        }
        for (int x = i + 2 * h; x < i + 3 * h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S3 += image[x][y];
            }
        }
        return (S1 - S2 + S3) * a / (3 * w * h);
    }
    else if (feature_type == "e")
    {
        double a = 4 * w * h;
        i = round_nearest_integer(i * e / 24);
        j = round_nearest_integer(j * e / 24);
        w = 0; // max(k for k in range(1, round_nearest_integer(1 + 2 * w * e / 24) // 2 + 1) if 2 * k <= e - j + 1)
        h = 0; // max(k for k in range(1, round_nearest_integer(1 + 2 * h * e / 24) // 2 + 1) if 2 * k <= e - i + 1)
        for (int k = 1; k < round_nearest_integer(1 + 2 * w * e / 24) / 2 + 1; k++)
        {
            if (2 * k <= e - j + 1)
            {
                w = k;
            }
        }
        for (int k = 1; k < round_nearest_integer(1 + 2 * h * e / 24) / 2 + 1; k++)
        {
            if (2 * k <= e - i + 1)
            {
                h = k;
            }
        }
        double S1 = 0;
        double S2 = 0;
        double S3 = 0;
        double S4 = 0;
        for (int x = i; x < i + h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S1 += image[x][y];
            }
        }
        for (int x = i + h; x < i + 2 * h; x++)
        {
            for (int y = j; y < j + w; y++)
            {
                S2 += image[x][y];
            }
        }
        for (int x = i; x < i + h; x++)
        {
            for (int y = j + w; y < j + 2 * w; y++)
            {
                S3 += image[x][y];
            }
        }
        for (int x = i + h; x < i + 2 * h; x++)
        {
            for (int y = j + w; y < j + 2 * w; y++)
            {
                S4 += image[x][y];
            }
        }
        return (S1 - S2 - S3 + S4) * a / (4 * w * h);
    }
    // Implement other feature types here

    else
    {
        throw invalid_argument("Unknown feature type");
    }
}

void print_time()
{
    static int x = 1;
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    std::cout << "time " << x << " : " << duration.count() << " s\n ";
    x++;
    start_time = end_time;
}

void update_learner(double W_pos_below, double W_neg_below, double W_pos_above, double W_neg_above, double tot_wights, double tau, double curr_M, Learner *cur_stump)
{
    double error_pos = W_pos_below + W_neg_above;
    double error_neg = W_neg_below + W_pos_above;
    int toggle = (error_pos <= error_neg) ? 1 : -1;
    double error = min(error_pos, error_neg) / tot_wights;
    if (error < cur_stump->error || (error == cur_stump->error && curr_M > cur_stump->margin))
    {
        cur_stump->error = error;
        cur_stump->threshold = tau;
        cur_stump->polarity = toggle;
        cur_stump->margin = curr_M;
    }
}

Learner *decision_stump(int **&X, int *&y, double *&weights, int feature_index, int *sorted_indices, int *X_sorted, int *Y_sorted, double *weights_sorted, double *&pos_weights_prefix, double *&neg_weights_prefix, pair<int, int> &dim)
{
    Learner *cur_stump = new Learner(0, 1, 2, 0, feature_index);
    int n = dim.first;
    for (int i = 0; i < n; i++)
    {
        sorted_indices[i] = i;
    }
    sort(sorted_indices, sorted_indices + n, [&](int i, int j)
         { return X[i][feature_index] < X[j][feature_index]; });

    for (int i = 0; i < n; i++)
    {
        X_sorted[i] = X[sorted_indices[i]][feature_index];
        Y_sorted[i] = y[sorted_indices[i]];
        weights_sorted[i] = weights[sorted_indices[i]];
    }
    for (int i = 0; i < n; i++)
    {
        if (Y_sorted[i] == 1)
        {
            pos_weights_prefix[i] = weights_sorted[i];
            neg_weights_prefix[i] = 0;
        }
        else
        {
            neg_weights_prefix[i] = weights_sorted[i];
            pos_weights_prefix[i] = 0;
        }
        if (i)
        {
            pos_weights_prefix[i] += pos_weights_prefix[i - 1];
            neg_weights_prefix[i] += neg_weights_prefix[i - 1];
        }
    }
    double tot_wights = pos_weights_prefix[n - 1] + neg_weights_prefix[n - 1];
    double tau = X_sorted[0] - 1;
    double W_pos_above = pos_weights_prefix[n - 1];
    double W_neg_above = neg_weights_prefix[n - 1];
    double W_pos_below = 0;
    double W_neg_below = 0;
    int curr_M = 1;
    int toggle = 1;

    for (int j = 0; j < n; j++)
    {
        update_learner(W_pos_below, W_neg_below, W_pos_above, W_neg_above, tot_wights, tau, curr_M, cur_stump);
        // cout << "for threshold " << cur_stump->threshold << " error is " << cur_stump->error << endl;
        while (true)
        {
            if (j + 1 < n && X_sorted[j] == X_sorted[j + 1])
                j++;
            else
                break;
        }
        if (j < n - 1)
        {
            tau = (X_sorted[j] + X_sorted[j + 1]) / 2;
            curr_M = X_sorted[j + 1] - X_sorted[j];
            W_pos_above = pos_weights_prefix[n - 1] - pos_weights_prefix[j];
            W_neg_above = neg_weights_prefix[n - 1] - neg_weights_prefix[j];
            W_pos_below = pos_weights_prefix[j];
            W_neg_below = neg_weights_prefix[j];
        }
        else
        {
            tau = X_sorted[j] + 1;
            curr_M = 1;
            W_pos_above = 0;
            W_neg_above = 0;
            W_pos_below = pos_weights_prefix[n - 1];
            W_neg_below = neg_weights_prefix[n - 1];
        }
    }
    update_learner(W_pos_below, W_neg_below, W_pos_above, W_neg_above, tot_wights, tau, curr_M, cur_stump);
    // cout << "for threshold " << cur_stump->threshold << " error is " << cur_stump->error << endl;

    return cur_stump;
}

// O(num_features * n)
Learner *best_stump(int **&X, int *&y, double *&weights, pair<int, int> &dim)
{

    int n = dim.first;
    double *pos_weights_prefix = new double[n];
    double *neg_weights_prefix = new double[n];

    int *sorted_indices = new int[n];
    int *X_sorted = new int[n];
    int *y_sorted = new int[n];
    double *weights_sorted = new double[n];
    Learner *best_stump = decision_stump(X, y, weights, 0, sorted_indices, X_sorted, y_sorted, weights_sorted, pos_weights_prefix, neg_weights_prefix, dim);

    for (int f = 0; f < dim.second; f++)
    {

        // num_features is around 160K , this part could be run on cuda and lunch too many threads here
        Learner *cur_stump = decision_stump(X, y, weights, f, sorted_indices, X_sorted, y_sorted, weights_sorted, pos_weights_prefix, neg_weights_prefix, dim);
        if (cur_stump->error < best_stump->error || (cur_stump->error == best_stump->error && cur_stump->margin > best_stump->margin))
        // if (cur_stump->error < best_stump->error)
        {
            delete best_stump;
            best_stump = cur_stump;
            best_stump->feature_index = f;
        }
        else
        {
            delete cur_stump;
        }
    }
    delete[] pos_weights_prefix;
    delete[] neg_weights_prefix;

    delete[] sorted_indices;
    delete[] X_sorted;
    delete[] y_sorted;
    delete[] weights_sorted;

    return best_stump;
}

int get_files(const std::string &path, string *&files, int num)
{
    // std::vector<std::string> files;
    int count = 0;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")
        {
            // files.push_back(entry.path().string());
            count++;
            if (num != -1 && count >= num)
            {
                break;
            }
        }
    }
    files = new string[count];
    count = 0;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")
        {
            files[count] = entry.path().string();
            count++;
            if (num != -1 && count >= num)
            {
                break;
            }
        }
    }
    return count;
}

int load_gray_images(const std::string &path, int ***&images, int num)
{
    string *files;
    int count = get_files(path, files, num);
    images = new int **[count];
    for (int i = 0; i < count; i++)
    {
        auto &file = files[i];
        int w, h, channels;
        unsigned char *img = stbi_load(file.c_str(), &w, &h, &channels, 0);
        if (img)
        {
            std::vector<int> grayscale_image(w * h);
            if (channels > 1)
                for (int i = 0; i < w * h; ++i)
                {
                    // Convert to grayscale assuming the image is in RGB format
                    grayscale_image[i] = (0.0 + img[i * channels] + img[i * channels + 1] + img[i * channels + 2]) / channels;
                }
            else
                for (int i = 0; i < w * h; ++i)
                {
                    grayscale_image[i] = img[i];
                }

            stbi_image_free(img);
            // Allocate a 2D vector to store the grayscale image data
            vector<vector<int>> imageVec(h, vector<int>(w));
            // Convert to grayscale and copy data from 1D array to 2D vector
            for (int i = 0; i < h; ++i)
            {
                for (int j = 0; j < w; ++j)
                {
                    int gray_index = (i * w + j) * channels;
                    imageVec[i][j] = grayscale_image[gray_index];
                }
            }
            images[i] = new int *[h];
            for (int j = 0; j < h; j++)
            {
                images[i][j] = new int[w];
                for (int k = 0; k < w; k++)
                {
                    images[i][j][k] = imageVec[j][k];
                }
            }
        }
    }
    delete[] files;
    return count;
}

// this function will be combination between load_gray_images and load_haar_like_features
// to help us avoid allocate memory for images and integral images

pair<int, int> load_features(const string &pos_path, const string &neg_path, int **&X, int *&Y, int pos_num, int neg_num)
{
    int count1 = 0, count2 = 0;
    string *files1;
    string *files2;

    count1 = get_files(pos_path, files1, pos_num);
    count2 = get_files(neg_path, files2, neg_num);

    int count = count1 + count2;
    int features_num = FEATURE_NUM;
    X = new int *[count];
    Y = new int[count];
    for (int i = 0; i < count1; i++)
        Y[i] = 1;
    for (int i = count1; i < count; i++)
        Y[i] = -1;

    int **imageVec = new int *[30];
    int **II = new int *[30];
    int *grayscale_image = new int[30 * 30];

    for (int i = 0; i < 30; i++)
    {
        imageVec[i] = new int[30];
    }
    for (int i = 0; i < 30; i++)
    {
        II[i] = new int[30];
    }

    for (int i = 0; i < count; i++)
    {
        string *file;
        if (i < count1)
            file = &files1[i];
        else
            file = &files2[i - count1];
        int w, h, channels;
        unsigned char *img = stbi_load(file->c_str(), &w, &h, &channels, 0);
        if (img)
        {
            if (channels > 1)
                for (int i = 0; i < w * h; ++i)
                {
                    grayscale_image[i] = (img[i * channels] + img[i * channels + 1] + img[i * channels + 2]) / channels;
                }
            else
                for (int i = 0; i < w * h; ++i)
                {
                    grayscale_image[i] = img[i];
                }
            stbi_image_free(img);
            // Convert to grayscale and copy data from 1D array to 2D vector
            for (int i = 0; i < h; ++i)
            {
                for (int j = 0; j < w; ++j)
                {
                    int gray_index = (i * w + j) * channels;
                    imageVec[i][j] = grayscale_image[gray_index];
                }
            }

            integral_image(imageVec, II, h, w);
            int *features;
            features_num = compute_haar_like_features(II, features);
            X[i] = features;
        }
    }
    delete[] files1;
    delete[] files2;
    delete[] grayscale_image;
    for (int i = 0; i < 30; i++)
    {
        delete[] imageVec[i];
        delete[] II[i];
    }
    delete[] imageVec;
    delete[] II;
    return make_pair(count, features_num);
}

matrices calc_acuracy_metrices(int *Y_test, int *predeictions, int count)
{
    // Initialize counters
    int true_positive = 0, true_negative = 0;
    int false_positive = 0, false_negative = 0;

    // Calculate the counts for TP, TN, FP, and FN
    for (size_t i = 0; i < count; ++i)
    {
        if (Y_test[i] == 1 && predeictions[i] == 1)
        {
            true_positive++;
        }
        else if (Y_test[i] == -1 && predeictions[i] == -1)
        {
            true_negative++;
        }
        else if (Y_test[i] == -1 && predeictions[i] == 1)
        {
            false_positive++;
        }
        else if (Y_test[i] == 1 && predeictions[i] == -1)
        {
            false_negative++;
        }
    }
    // Calculate accuracy, error rate, false positive rate, and false negative rate
    double accuracy = static_cast<double>(true_positive + true_negative) / count;
    double error_rate = static_cast<double>(false_positive + false_negative) / count;
    int total_positives = true_positive + false_negative;
    int total_negatives = true_negative + false_positive;
    double false_positive_rate = total_negatives > 0 ? static_cast<double>(false_positive) / total_negatives : 0;
    double false_negative_rate = total_positives > 0 ? static_cast<double>(false_negative) / total_positives : 0;

    matrices m;
    m.accuracy = accuracy;
    m.error_rate = error_rate;
    m.false_positive_rate = false_positive_rate;
    m.false_negative_rate = false_negative_rate;
    return m;
}