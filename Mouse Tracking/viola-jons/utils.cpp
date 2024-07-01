#include "utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "learner.h"
#include <chrono>
#include <string>
using namespace std;
// using namespace std::chrono;
const int step_size = 4;
const int positive_pretection = 1;
const int negative_pretection = -1;
const double err = 1e-6;
auto start_time = std::chrono::high_resolution_clock::now();
auto end_time = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration;
std::chrono::duration<double> duration2;
std::chrono::duration<double> duration3;
std::chrono::duration<double> duration4;
std::chrono::duration<double> duration5;
std::chrono::duration<double> duration6;
std::chrono::duration<double> duration7;
std::chrono::duration<double> duration8;

void integral_image(const vector<vector<double>> &I, vector<vector<double>> &II)
{
    int N = I.size();
    int M = I[0].size();
    II.resize(N, vector<double>(M));

    // Set II(1, 1) = I(1, 1)
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

int sum_region(const vector<vector<double>> &ii, int x1, int y1, int x2, int y2)
{
    int A = (x1 > 0 && y1 > 0) ? ii[x1 - 1][y1 - 1] : 0;
    int B = (x1 > 0) ? ii[x1 - 1][y2] : 0;
    int C = (y1 > 0) ? ii[x2][y1 - 1] : 0;
    int D = ii[x2][y2];
    return D - B - C + A;
}

vector<double> compute_haar_like_features(const vector<vector<double>> &img, const vector<vector<double>> &II)
{
    // assert(img.size() == 24 && img[0].size() == 24);

    vector<double> features;
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
                    features.push_back(S1 - S2);
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
                    features.push_back(S1 - S2 + S3);
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
                    features.push_back(S1 - S2);
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
                    features.push_back(S1 - S2 + S3);
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
                    features.push_back(S1 - S2 - S3 + S4);
                    f++;
                }
            }
        }
    }

    return features;
}

double haar_feature_scaling(const vector<vector<double>> &image, const string &feature_type, int i, int j, int w, int h)
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
Learner *decision_stump(vector<vector<double>> &X, const vector<int> &y, const vector<double> &weights, int feature_index, vector<int> &sorted_indices, vector<vector<double> *> &X_sorted, vector<int> &y_sorted, vector<double> &weights_sorted)
{
    int n = y.size();
    for (int i = 0; i < n; i++)
    {
        sorted_indices[i] = i;
    }
    sort(sorted_indices.begin(), sorted_indices.end(), [&](int i, int j)
         { return X[i][feature_index] < X[j][feature_index]; });
    for (int i = 0; i < n; i++)
    {
        X_sorted[i] = &(X[sorted_indices[i]]);
        y_sorted[i] = y[sorted_indices[i]];
        weights_sorted[i] = weights[sorted_indices[i]];
    }
    double tau = (*X_sorted[0])[feature_index] - 1;
    double W_pos_above = 0;
    double W_neg_above = 0;
    for (int i = 0; i < n; i++)
    {
        if (y_sorted[i] == 1)
        {
            W_pos_above += weights_sorted[i];
        }
        else
        {
            W_neg_above += weights_sorted[i];
        }
    }
    double W_pos_below = 0;
    double W_neg_below = 0;
    int curr_M = 0;
    int toggle = 1;

    Learner *cur_stump = new Learner(0, 1, 2, 0, 0);
    for (int j = 0; j < n; j++)
    {
        double error_pos = W_neg_above + W_pos_below;
        double error_neg = W_pos_above + W_neg_below;
        toggle = (error_pos <= error_neg) ? 1 : -1;
        double error = min(error_pos, error_neg);
        if (error < cur_stump->error || (error == cur_stump->error && curr_M > cur_stump->margin))
        {
            cur_stump->error = error;
            cur_stump->threshold = tau;
            cur_stump->polarity = toggle;
            cur_stump->margin = curr_M;
        }
        while (true)
        {
            if (y_sorted[j] == -1)
            {
                W_neg_below += weights_sorted[j];
                W_neg_above -= weights_sorted[j];
            }

            else
            {
                W_pos_below += weights_sorted[j];
                W_pos_above -= weights_sorted[j];
            }

            if (j + 1 < n && (*X_sorted[j])[feature_index] == (*X_sorted[j + 1])[feature_index])
                j++;
            else
                break;
        }
        if (j < n - 1)
        {
            tau = ((*X_sorted[j])[feature_index] + (*X_sorted[j + 1])[feature_index]) / 2;
            curr_M = (*X_sorted[j + 1])[feature_index] - (*X_sorted[j])[feature_index];
        }
    }

    return cur_stump;
}

// O(num_features * n)
Learner *best_stump(vector<vector<double>> &X, const vector<int> &y, const vector<double> &weights, int num_features)
{

    int n = X.size();
    vector<int> sorted_indices(n);
    vector<vector<double> *> X_sorted(n, nullptr);
    vector<int> y_sorted(n);
    vector<double> weights_sorted(n);
    Learner *best_stump = decision_stump(X, y, weights, 0, sorted_indices, X_sorted, y_sorted, weights_sorted);
    start_time = std::chrono::high_resolution_clock::now();
    for (int f = 0; f < num_features; f++)
    {
        // if (f % 1000 == 0)
        // {
        //     duration = end_time - start_time;
        //     cout << "feature number " << f << endl;
        //     std::cout << "single design stump time: " << duration.count() << " s\n";

        //     start_time = chrono::high_resolution_clock::now();
        //     int x = 0;
        //     for (int i = 0; i < 1000; i++)
        //     {
        //         x++;
        //     }
        //     end_time = chrono::high_resolution_clock::now();
        //     duration = end_time - start_time;
        //     cout << "time for 1000 iterations : " << duration.count() << " s\n";
        // }

        // num_features is around 160K , this part could be run on cuda and lunch too many threads here
        // start_time = std::chrono::high_resolution_clock::now();
        Learner *cur_stump = decision_stump(X, y, weights, f, sorted_indices, X_sorted, y_sorted, weights_sorted);
        // end_time = std::chrono::high_resolution_clock::now();
        if (cur_stump->error < best_stump->error || (cur_stump->error == best_stump->error && cur_stump->margin > best_stump->margin))
        {
            best_stump = cur_stump;
            best_stump->feature_index = f;
        }
        else
        {
            delete cur_stump;
        }
    }
    return best_stump;
}
