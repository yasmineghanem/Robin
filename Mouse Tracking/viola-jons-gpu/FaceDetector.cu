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
    matrices mat;
    for (int i = 0; i < this->train_dim.first; i++)
    {
        predictions[i] = fl->predict(X_train[i], sl);
    }
    mat = calc_acuracy_metrices(y_train, predictions, this->train_dim.first);
    for (int i = 0; i < get<0>(this->val_dim); i++)
    {
        predictions[i] = fl->predict(X_val[i], get<1>(this->val_dim), sl);
    }
    matrices val_mat = calc_acuracy_metrices(y_val, predictions, get<0>(this->val_dim));
    mat.false_positive_rate = max(val_mat.false_positive_rate, mat.false_positive_rate);
    mat.false_negative_rate = max(val_mat.false_negative_rate, mat.false_negative_rate);

    return mat;
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
void FaceDetector::train(double Yo, double Yl, double Bl)
{
    double cur_Y = 1;
    int l = 0;
    int *predictions = new int[max(this->train_dim.first, get<0>(this->val_dim))];
    int last = 0;
    while (cur_Y > Yo)
    {
        double u = 1e-2;
        l += 1;
        int Nl = min(10 * l + 10, 200);
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
                    sl = -1.0;
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

void FaceDetector::process(int **&img, int ***&color_img, int M, int N, double c)
{
    int L = this->cascade.size();
    vector<window *> P;
    int **II = new int *[M];
    long long **IIsq = new long long *[M];

    for (int i = 0; i < M; ++i)
    {
        II[i] = new int[N];
        IIsq[i] = new long long[N];
        for (int j = 0; j < N; ++j)
        {
            IIsq[i][j] = img[i][j] * img[i][j];
        }
    }

    integral_image(img, II, M, N);
    integral_image(IIsq, IIsq, M, N);
    // Set initial windows
    long long tot = 0;
    for (int e = static_cast<int>(24 * c); e <= std::min(M, N); e = static_cast<int>(e * c))
    {
        for (int i = 0; i + e < M; ++i)
        {
            for (int j = 0; j + e < N; ++j)
            {
                window *w = new window;
                w->x = i;
                w->y = j;
                w->w = e;
                w->h = e;
                P.push_back(w);
                tot += (long long)e * e;
            }
        }
    }
    std::cout << "P size before filtter: " << P.size() << endl;
    std::cout << "total pixels of windows : " << tot << endl;
    // // Cascade layers
    for (int k = 0; k < P.size(); k++)
    {
        int i = P[k]->x;
        int j = P[k]->y;
        int e = P[k]->w;
    }

    // long long sum = II[M - 1][N - 1];
    // long long sq_sum = IIsq[M - 1][N - 1];
    // double mean = (double)sum / (M * N);
    // double variance = ((long double)sq_sum / (M * N)) - (mean * mean);
    // double stddev = std::sqrt(variance);
    // for (int x = 0; x < M; ++x)
    // {
    //     for (int y = 0; y < N; ++y)
    //     {
    //         img[x][y] = (img[x][y] - mean) / stddev;
    //     }
    // }

    std::vector<window *> newP;
    for (const auto &win : P)
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
                windowImg[x] = &img[i + x][j];
            }
            int prediction = this->predict(windowImg, e);
            if (prediction == 1)
                newP.push_back(win);
            else
                delete win;

            delete[] windowImg;
        }
    }
    P = newP;
    tot = 0;
    for (int i = 0; i < P.size(); i++)
    {
        tot += P[i]->w * P[i]->h;
    }
    cout << "P size after filtter: " << P.size() << endl;
    cout << "total pixels of windows after filtter: " << tot << endl;

    drawGreenRectangles(color_img, M, N, P);
    for (int i = 0; i < M; ++i)
    {
        delete[] II[i];
        delete[] IIsq[i];
    }
    delete[] II;
    delete[] IIsq;
}

FaceDetector::~FaceDetector()
{
    for (auto &fl : cascade)
    {
        delete fl;
    }
}
