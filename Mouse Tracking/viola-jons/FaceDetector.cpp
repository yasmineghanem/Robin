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
FaceDetector::FaceDetector(vector<vector<int>> &X_train, vector<int> &y_train, vector<vector<vector<int>>> &X_val, vector<int> &y_val)
{
    this->X_train = X_train;
    this->y_train = y_train;
    this->X_val = X_val;
    this->y_val = y_val;
    // for (auto &x : X_val)
    // {
    //     integral_image(x, x);
    // }
    this->n_train = X_train.size();
    this->n_validation = X_val.size();
}
FaceDetector::FaceDetector()
{
}
matrices FaceDetector::evaluate_single_layer(AdaBoost &fl, vector<int> &predictions, double sl)
{
    // set the value of the shifted value sl
    matrices mat;
    for (int i = 0; i < min(this->n_train, (int)predictions.size()); i++)
    {
        predictions[i] = fl.predict(X_train[i], sl);
    }
    mat = calc_acuracy_metrices(y_train, predictions);
    // TODO get the metrices for the validation set without pre calculate all the haar like features
    // maximize with the result from the validation
    return mat;
}
void FaceDetector::remove_false_data()
{
    // TODO remove from the validation also
    // TODO dont create new vectors , swap the values and change ths size

    // remove the false negatives and true negatives detected by the current cascade
    vector<vector<int>> new_X_train;
    vector<int> new_y_train;
    int l = this->cascade.size() - 1;
    for (int i = 0; i < this->n_train; i++)
    {
        if (this->cascade[l].predict(this->X_train[i], this->shif[l]) == 1)
        {
            new_X_train.push_back(this->X_train[i]);
            new_y_train.push_back(this->y_train[i]);
        }
    }
    this->X_train = new_X_train;
    this->y_train = new_y_train;
    this->n_train = new_X_train.size();
}
void FaceDetector::train(double Yo, double Yl, double Bl)
{
    double cur_Y = 1;
    int l = 0;
    vector<int> predictions(max(this->n_train, this->n_validation));
    int last = 0;
    while (cur_Y > Yo)
    {
        double u = 1e-2;
        l += 1;
        int Nl = min(10 * l + 10, 200);
        double sl = 0;
        int Tl = 1;
        AdaBoost fl(this->X_train, this->y_train);
        fl.train(1);
        double B, Y;
        while (true)
        {
            bool train_again = false;
            auto mat = this->evaluate_single_layer(fl, predictions, sl);
            Y = mat.false_positive_rate, B = mat.false_negative_rate;
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
                    sl = -1;
                    auto mat = this->evaluate_single_layer(fl, predictions, sl);
                    double Y = mat.false_positive_rate, B = mat.false_negative_rate;
                    while (B > Bl)
                    {
                        sl += 0.05;
                        mat = this->evaluate_single_layer(fl, predictions, sl);
                        B = mat.false_negative_rate;
                        Y = mat.false_positive_rate;
                    }
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
                fl.train(1);
        }
        //  Remove the false negatives and true negatives detected by the current casca
        cascade.push_back(fl);
        shif.push_back(sl);
        this->remove_false_data();
        cout << "layer " << l << " is trained" << endl;
        cout << "false positive rate: " << Y << endl;
        cout << "false negative rate: " << B << endl;
        this->save("face1");
    }
}
int FaceDetector::predict(vector<vector<int>> &img)
{

    for (int i = 0; i < this->cascade.size(); i++)
    {
        // TODO implement method to predict using the needed features only not calc all the features
        // if cascade fine it is negative then return
        if (this->cascade[i].predict(compute_haar_like_features(img), this->shif[i]) == -1)
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
        cascade[i].saveAsText(folder + "/model" + to_string(i) + ".txt");
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
        cascade[i].loadFromText(folder + "/model" + to_string(i) + ".txt");
    }
}
