#include "FaceDetector.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "AdaBoost.h"
#include "learner.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <cmath>

using namespace std;
FaceDetector::FaceDetector(int **&X_train, int *&y_train, int ***&X_val, int *&y_val, pair<int, int> train_dim, tuple<int, int, int> val_dim, string folder)
{
    this->X_train = X_train;
    this->y_train = y_train;
    this->X_val = X_val;
    this->y_val = y_val;
    this->train_dim = train_dim;
    this->val_dim = val_dim;
    this->folder = folder;
}

FaceDetector::FaceDetector()
{
}

matrices FaceDetector::evaluate_single_layer(AdaBoost *fl, int *&predictions, double sl)
{
    // set the value of the shifted value sl

    for (int i = 0; i < this->train_dim.first; i++)
    {
        predictions[i] = fl->predict(X_train[i], sl);
    }
    matrices train_mat = calc_acuracy_metrices(y_train, predictions, this->train_dim.first);
    int p = 0, neg = 0;
    for (int i = 0; i < get<0>(this->val_dim); i++)
    {
        predictions[i] = fl->predict(X_val[i], get<1>(this->val_dim), sl);
    }
    matrices val_mat = calc_acuracy_metrices(y_val, predictions, get<0>(this->val_dim));
    train_mat.false_positive_rate = max(val_mat.false_positive_rate, train_mat.false_positive_rate);
    train_mat.false_negative_rate = max(val_mat.false_negative_rate, train_mat.false_negative_rate);

    return train_mat;
}

void FaceDetector::remove_negative_train_data()
{
    // remove the false negatives and true negatives detected by the current cascade
    int l = this->cascade.size() - 1;
    int new_count = 0;
    int index = -1;

    for (int i = 0; i < this->train_dim.first; i++)
    {
        if (this->cascade[l]->predict(this->X_train[i], this->shif[l]) == 1)
        {
            new_count++;
            if (index != -1)
            {
                swap(this->X_train[i], this->X_train[index]);
                swap(this->y_train[i], this->y_train[index]);
                index++;
            }
        }
        else if (index == -1)
        {
            index = i;
        }
    }
    this->train_dim.first = new_count;
}

void FaceDetector::remove_negative_val_data()
{
    // remove the false negatives and true negatives detected by the current cascade
    int l = this->cascade.size() - 1;
    int new_count = 0;
    int index = -1;
    for (int i = 0; i < get<0>(this->val_dim); i++)
    {
        if (this->cascade[l]->predict(this->X_val[i], get<1>(this->val_dim), this->shif[l]) == 1)
        {
            new_count++;
            if (index != -1)
            {
                swap(this->X_val[i], this->X_val[index]);
                swap(this->y_val[i], this->y_val[index]);
                index++;
            }
        }
        else if (index == -1)
        {
            index = i;
        }
    }
    get<0>(this->val_dim) = new_count;
}

void FaceDetector::train(double Yo, double Yl, double Bl, bool restric)
{
    // double cur_Y = 1;
    int l = 0;
    int *predictions = new int[max(this->train_dim.first, get<0>(this->val_dim))];
    int last = 0;
    while (cur_Y > Yo)
    {
        double u = 1e-2;
        l += 1;

        int Nl = min(10 * l + 10, 200);
        if (restric)
        {
            Nl = 30;
        }
        double sl = 0;
        int Tl = 1;
        AdaBoost *fl = new AdaBoost(this->X_train, this->y_train, this->train_dim);
        fl->train(1);
        double B, Y;
        while (true)
        {
            bool train_again = false;
            auto mat = this->evaluate_single_layer(fl, predictions, sl);
            Y = mat.false_positive_rate, B = mat.false_negative_rate;
            cout << "adaboost number :  " << l << ", layer number : " << Tl << ", shift: " << sl << " false positive rate : " << Y << ", false negative rate : " << B << endl;
            if (Y <= Yl && B <= Bl)
            {
                cur_Y = cur_Y * Y;
                break;
            }
            else if (Y <= Yl && B > Bl && u > 1e-5)
            {

                sl += u;
                if (last == -1)
                {
                    u = u / 2;
                    sl -= u;
                }
                last = 1;
            }
            else if (Y > Yl && B <= Bl && u > 1e-5)
            {

                sl -= u;
                if (last == 1)
                {
                    u = u / 2;
                    sl += u;
                }
                last = -1;
            }
            else
            {

                if (Tl > Nl)
                {

                    // the shift sl is set to the smallest value that satisfies the false negative requirement.
                    sl = -1.1;
                    auto mat = this->evaluate_single_layer(fl, predictions, sl);
                    Y = mat.false_positive_rate, B = mat.false_negative_rate;
                    cout << "adaboost number :  " << l << " layer number : " << Tl << " entered the dead loop" << endl;
                    while (B > Bl && sl < 1.0)
                    {
                        sl += 0.01;
                        mat = this->evaluate_single_layer(fl, predictions, sl);
                        B = mat.false_negative_rate;
                        Y = mat.false_positive_rate;
                        cout << " shift: " << sl << " false positive rate : " << Y << " false negative rate : " << B << endl;
                    }
                    cout << "adaboost number :  " << l << " layer number : " << Tl << " go out from the dead loop" << endl;

                    cur_Y = cur_Y * Y;
                    break;
                }
                else
                {

                    Tl += 1;
                    train_again = true;
                }
            }
            if (train_again)
            {
                fl->train(1);
            }
        }

        //  Remove the false negatives and true negatives detected by the current casca
        cascade.push_back(fl);
        shif.push_back(sl);
        cout << " training size before removing: " << this->train_dim.first << endl;
        cout << " validation size before removing: " << get<0>(this->val_dim) << endl;
        this->remove_negative_train_data();
        this->remove_negative_val_data();
        cout << " training size after removing: " << this->train_dim.first << endl;
        cout << " validation size after removing: " << get<0>(this->val_dim) << endl;

        cout << "layer " << l << " is trained" << endl;
        cout << "false positive rate: " << Y << endl;
        cout << "false negative rate: " << B << endl;
        this->save(folder);
    }
}

int FaceDetector::predict(int **&img, int size, double devide)
{

    for (int i = 0; i < this->cascade.size(); i++)
    {
        // if cascade fined it is negative then return
        if (this->cascade[i]->predict(img, size, this->shif[i], devide) == -1)
        {
            return -1;
        }
    }
    return 1;
}

bool FaceDetector::window_test1(window *win, int **II, long long **IIsq)
{
    int i = win->x;
    int j = win->y;
    int e = win->w;
    long long sum = 0, sq_sum = 0;
    sum = sum_region(II, i, j, i + e, j + e);
    sq_sum = sum_region(IIsq, i, j, i + e, j + e);
    double mean = (double)sum / (e * e);
    double variance = ((long double)sq_sum / (e * e)) - (mean * mean);
    double stddev = std::sqrt(variance);
    if (stddev > 1)
    {
        int **windowImg = new int *[e];
        for (int x = 0; x < e; ++x)
        {
            windowImg[x] = &II[i + x][j];
        }
        int prediction = this->predict(windowImg, e);
        delete[] windowImg;

        if (prediction == 1)
            return true;
    }
    return false;
}
// Utility function to check if a point (px, py) is inside a window
bool isInside(const window &win, int px, int py)
{
    return (px >= win.x && px < win.x + win.w && py >= win.y && py < win.y + win.h);
}
// Function to calculate the center of a window
pair<int, int> getCenter(const window &win)
{
    return {win.x + win.w / 2, win.y + win.h / 2};
}

void FaceDetector::window_test2(vector<window *> &windows, int **&img, int M, int N)
{

    // Create an M × N matrix E filled with zeros
    int **E = new int *[M];
    // int **freq = new int *[M];
    for (int i = 0; i < M; ++i)
    {
        E[i] = new int[N]{0};
        // freq[i] = new int[N]{0};
    }
    // Fill E with the sizes of the windows
    for (const auto &w : windows)
    {
        E[w->x][w->y] = max(w->w, E[w->x][w->y]);
        // E[w->x][w->y] = w->w;
        // freq[w->x][w->y]++;
    }
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         if (freq[i][j] > 0)
    //         {
    //             E[i][j] /= freq[i][j];
    //         }
    //     }
    // }
    // Run a connected component algorithm on E
    vector<vector<int>> labels(M, vector<int>(N, 0));
    int current_label = 0;
    vector<vector<pair<int, int>>> components;

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (E[i][j] > 0 && labels[i][j] == 0)
            {
                // start dfs from this point
                current_label++;
                vector<pair<int, int>> stack = {{i, j}};
                vector<pair<int, int>> component;

                while (!stack.empty())
                {
                    pair<int, int> p = stack.back();
                    stack.pop_back();
                    int x = p.first;
                    int y = p.second;

                    if (x < 0 || x >= M || y < 0 || y >= N || labels[x][y] > 0 || E[x][y] != E[i][j])
                    // if (x < 0 || x >= M || y < 0 || y >= N || labels[x][y] > 0)
                    {
                        continue;
                    }

                    labels[x][y] = current_label;
                    component.push_back({x, y});
                    stack.push_back({x + this->stride, y});
                    stack.push_back({x - this->stride, y});
                    stack.push_back({x, y + this->stride});
                    stack.push_back({x, y - this->stride});
                }

                components.push_back(component);
            }
        }
    }

    // Process each component
    vector<pair<window, double>> P;
    for (const auto &component : components)
    {
        int eC = E[component[0].first][component[0].second];
        int size = component.size();
        double confidence = size * (24.0 / eC);

        if (confidence >= min_confidence)
        {
            int max_x = component[0].first, max_y = component[0].second;
            int avg_x = 0, avg_y = 0;
            for (const auto &p : component)
            {
                int x = p.first;
                int y = p.second;
                max_x = max(max_x, x);
                max_y = max(max_y, y);
                avg_x += x;
                avg_y += y;
            }
            avg_x /= size;
            avg_y /= size;
            P.push_back({{avg_x, avg_y, eC, eC}, confidence});
        }
    }

    //  Sort the elements in P in ascending order of window size
    sort(P.begin(), P.end(), [](const pair<window, double> &a, const pair<window, double> &b)
         { return (a.first.w * a.first.h) < (b.first.w * b.first.h); });

    // Remove redundant windows
    for (size_t i = 0; i < P.size(); ++i)
    {
        auto center = getCenter(P[i].first);
        for (size_t j = i + 1; j < P.size(); ++j)
        {

            if (isInside(P[j].first, center.first, center.second))
            {
                // window i has a higher detection confidence than window j ,detlet window j
                if (P[i].second > P[j].second)
                {
                    P[j].first = {0, 0, 0, 0}; // Remove P[j]
                }
                else
                {
                    P[i].first = {0, 0, 0, 0}; // Remove P[i]
                    break;
                }
            }
        }
    }

    for (auto &w : windows)
    {
        delete w;
    }
    windows.clear();
    for (auto &w : P)
    {
        if (w.first.w == 0 || w.first.h == 0)
            continue;
        window *x = new window;
        x->x = w.first.x;
        x->y = w.first.y;
        x->w = w.first.w;
        x->h = w.first.h;
        windows.push_back(x);
    }

    // Clean up the allocated memory for E
    for (int i = 0; i < M; ++i)
    {
        delete[] E[i];
    }
    delete[] E;
}

inline bool skin_pixel(int r, int g, int b)
{
    return r > g && r > b;
}

bool skin_test(int **skin_denisty, int size, int i, int j)
{
    const double threshold = 0.7;
    int skin = sum_region(skin_denisty, i, j, i + size, j + size);
    return (double)skin / (size * size) >= threshold;
}

void FaceDetector::window_test3(vector<window *> &windows, int **&img, int M, int N)
{
    // Create an M × N matrix E filled with zeros
    int **E = new int *[M];
    int **freq = new int *[M];
    for (int i = 0; i < M; ++i)
    {
        E[i] = new int[N]{0};
        freq[i] = new int[N]{0};
    }

    // Fill E with the sizes of the windows
    for (const auto &w : windows)
    {
        E[w->x][w->y] = w->w;
        freq[w->x][w->y]++;
    }

    // Run a connected component algorithm on E
    vector<vector<int>> labels(M, vector<int>(N, 0));
    int current_label = 0;
    vector<vector<pair<int, int>>> components;

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (freq[i][j] > 0)
            {
                components.push_back({{i, j}});
            }
            // continue;
            if (E[i][j] > 0 && labels[i][j] == 0)
            {
                // start dfs from this point
                current_label++;
                vector<pair<int, int>> stack = {{i, j}};
                vector<pair<int, int>> component;

                while (!stack.empty())
                {
                    pair<int, int> p = stack.back();
                    stack.pop_back();
                    int x = p.first;
                    int y = p.second;

                    if (x < 0 || x >= M || y < 0 || y >= N || labels[x][y] > 0)
                    {
                        continue;
                    }

                    labels[x][y] = current_label;
                    component.push_back({x, y});
                    stack.push_back({x + 3, y});
                    stack.push_back({x - 3, y});
                    stack.push_back({x, y + 1});
                    stack.push_back({x, y - 1});
                }

                components.push_back(component);
            }
        }
    }

    // Process each component
    vector<pair<window, double>> P;
    for (const auto &component : components)
    {
        int eC = E[component[0].first][component[0].second];
        int size = freq[component[0].first][component[0].second];
        double confidence = size * (24.0 / eC);

        if (confidence >= min_confidence)
        {
            int max_x = 0, max_y = 0;
            for (const auto &p : component)
            {
                int x = p.first;
                int y = p.second;
                max_x = max(max_x, x);
                max_y = max(max_y, y);
            }
            P.push_back({{max_x, max_y, eC, eC}, confidence});
        }
    }

    //  Sort the elements in P in ascending order of window size
    sort(P.begin(), P.end(), [](const pair<window, double> &a, const pair<window, double> &b)
         { return (a.first.w * a.first.h) < (b.first.w * b.first.h); });

    //  Remove redundant windows
    // for (size_t i = 0; i < P.size(); ++i)
    // {
    //     auto center = getCenter(P[i].first);
    //     for (size_t j = i + 1; j < P.size(); ++j)
    //     {

    //         if (isInside(P[j].first, center.first, center.second))
    //         {
    //             // window i has a higher detection confidence than window j ,detlet window j
    //             if (P[i].second > P[j].second)
    //             {
    //                 P[j].first = {0, 0, 0, 0}; // Remove P[j]
    //             }
    //             else
    //             {
    //                 P[i].first = {0, 0, 0, 0}; // Remove P[i]
    //                 break;
    //             }
    //         }
    //     }
    // }

    for (auto &w : windows)
    {
        delete w;
    }
    windows.clear();
    for (auto &w : P)
    {
        if (w.first.w == 0 || w.first.h == 0)
            continue;
        window *x = new window;
        x->x = w.first.x;
        x->y = w.first.y;
        x->w = w.first.w;
        x->h = w.first.h;
        windows.push_back(x);
    }

    // Clean up the allocated memory for E
    for (int i = 0; i < M; ++i)
    {
        delete[] E[i];
    }
    delete[] E;
}

vector<window *> FaceDetector::process(int **&img, int ***&color_img, int M, int N, double c)
{
    const bool return_biggest = 1;
    int L = this->cascade.size();
    int **II = new int *[M];
    long long **IIsq = new long long *[M];
    int **skin_denisty = new int *[M];
    for (int i = 0; i < M; ++i)
    {
        skin_denisty[i] = new int[N];
        II[i] = new int[N];
        IIsq[i] = new long long[N];
        for (int j = 0; j < N; ++j)
        {
            IIsq[i][j] = img[i][j] * img[i][j];
            skin_denisty[i][j] = skin_pixel(color_img[i][j][0], color_img[i][j][1], color_img[i][j][2]);
        }
    }
    integral_image(skin_denisty, skin_denisty, M, N);
    integral_image(img, II, M, N);
    integral_image(IIsq, IIsq, M, N);
    vector<window *> P;

    vector<int> sizes;
    int last = 24;
    while (last <= min(M, N))
    {
        if (last * last >= this->smallest_box * M * N)
            sizes.push_back(last);
        last = static_cast<int>(last * c);
    }
    std::mutex mtx;

    auto worker = [&](int start, int end)
    {
        vector<window *> my_P;
        window my_temp;
        for (int j = start; j < end; ++j)
        {
            for (int i = 0; i + sizes[j] < M; i += this->stride)
            {
                for (int k = 0; k + sizes[j] < N; k += this->stride)
                {
                    if (!skin_test(skin_denisty, sizes[j], i, k))
                        continue;
                    my_temp.x = i;
                    my_temp.y = k;
                    my_temp.w = sizes[j];
                    my_temp.h = sizes[j];
                    if (this->window_test1(&my_temp, II, IIsq))
                    {
                        window *w = new window;
                        w->x = i;
                        w->y = k;
                        w->w = sizes[j];
                        w->h = sizes[j];
                        std::lock_guard<std::mutex> lock(mtx);
                        P.push_back(w);
                    }
                }
            }
        }
    };
    size_t num_threads = max(2, min((int)sizes.size(), (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    int sizes_per_thread = (sizes.size()) / num_threads;
    int remaining = sizes.size() % num_threads;
    for (size_t i = 0; i < num_threads; ++i)
    {
        int start = i * sizes_per_thread;
        int end = start + sizes_per_thread + (i < remaining ? 1 : 0);
        threads.emplace_back(worker, start, end);
    }
    for (auto &t : threads)
    {
        t.join();
    }

    std::cout << "P size after filtter1 : " << P.size() << endl;
    window_test2(P, img, M, N);
    std::cout << "P size after filtter2 : " << P.size() << endl;
    // drawGreenRectangles(color_img, M, N, P, 1);

    for (int i = 0; i < M; ++i)
    {
        delete[] II[i];
        delete[] IIsq[i];
        delete[] skin_denisty[i];
    }
    delete[] II;
    delete[] IIsq;
    delete[] skin_denisty;
    return P;
}

void FaceDetector::rebuild()
{
    vector<AdaBoost *> new_cascade;
    vector<double> new_shif;
    int index = 0;
    for (auto &fl : cascade)
    {
        new_cascade.push_back(fl);
        new_shif.push_back(shif[index]);
        index++;
    }
    cascade.clear();
    shif.clear();
    int *predictions = new int[max(this->train_dim.first, get<0>(this->val_dim))];

    for (int i = 0; i < new_cascade.size(); i++)
    {
        cascade.push_back(new_cascade[i]);
        shif.push_back(new_shif[i]);
        auto mat = this->evaluate_single_layer(cascade[i], predictions, shif[i]);
        double Y = mat.false_positive_rate, B = mat.false_negative_rate;
        cur_Y = cur_Y * Y;
        cout << "layer " << i << " is trained" << endl;
        cout << "false positive rate: " << Y << endl;
        cout << "false negative rate: " << B << endl;
        cout << " training size before removing: " << this->train_dim.first << endl;
        cout << " validation size before removing: " << get<0>(this->val_dim) << endl;
        this->remove_negative_train_data();
        this->remove_negative_val_data();
        cout << " training size after removing: " << this->train_dim.first << endl;
        cout << " validation size after removing: " << get<0>(this->val_dim) << endl;
        cout << "--------------------------------------" << endl;
    }
}
void FaceDetector::save(const string folder)
{
    // Create directory if it doesn't exist
    std::filesystem::create_directories(folder);

    // Open meta.txt for writing
    ofstream metaFile(folder + "/meta.txt");
    if (!metaFile.is_open())
    {
        cerr << "Failed to open meta.txt for writing" << endl;
        return;
    }

    // Write the size of the cascade
    metaFile << cascade.size() << endl;

    // Write the values of the shif vector
    for (const auto &s : shif)
    {
        metaFile << s << endl;
    }
    metaFile.close();

    // Save each layer of the cascade
    for (size_t i = 0; i < cascade.size(); ++i)
    {
        cascade[i]->saveAsText(folder + "/model" + to_string(i) + ".txt");
    }
}

void FaceDetector::load(const string folder)
{
    // Open meta.txt for reading
    ifstream metaFile(folder + "/meta.txt");
    if (!metaFile.is_open())
    {
        cerr << "Failed to open meta.txt for reading" << endl;
        return;
    }

    // Read the size of the cascade
    size_t cascadeSize;
    metaFile >> cascadeSize;

    // Read the values of the shif vector
    shif.clear();
    double value;
    for (size_t i = 0; i < cascadeSize; ++i)
    {
        metaFile >> value;
        shif.push_back(value);
    }
    metaFile.close();

    // Load each layer of the cascade
    cascade.clear();
    cascade.resize(cascadeSize);
    for (size_t i = 0; i < cascadeSize; ++i)
    {
        cascade[i] = new AdaBoost();
        cascade[i]->loadFromText(folder + "/model" + to_string(i) + ".txt");
    }
}

FaceDetector::~FaceDetector()
{
    for (auto &fl : cascade)
    {
        delete fl;
    }
}
