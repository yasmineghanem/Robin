#include "AdaBoost.h"
#include "learner.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <fstream>
#include "const.h"
#include <vector>

using namespace std;
const double err = 1e-6;
AdaBoost::~AdaBoost()
{
    for (size_t i = 0; i < learners.size(); ++i)
    {
        delete learners[i];
    }
}
AdaBoost::AdaBoost(int **&X, int *&y, pair<int, int> dim) : X(X), y(y), train_dim(dim)
{
    // intialize the weights
    int n_pos = 0;
    int n_neg = 0;
    for (size_t i = 0; i < this->train_dim.first; ++i)
    {
        if (y[i] == 1)
            n_pos++;
        else
            n_neg++;
    }
    this->weights = new double[this->train_dim.first];
    for (size_t i = 0; i < this->train_dim.first; ++i)
    {
        if (y[i] == 1)
            weights[i] = (0.5 / n_pos);
        else
            weights[i] = (0.5 / n_neg);
    }
}
AdaBoost::AdaBoost()
{
}

void AdaBoost::train(int T)
{
    int n = this->train_dim.first;
    for (int t = 0; t < T; t++)
    {

        // find the best learner
        Learner *learner = best_stump_threads(this->X, this->y, this->weights, this->train_dim);
        // compute the error
        double error = learner->error;
        learners.push_back(learner);
        if (abs(error - 0) < err)
        {
            alphas.push_back(1);
            break;
        }

        // compute the alpha
        double alpha = 0.5 * log((1 - error) / error);
        alphas.push_back(alpha);

        // update the weights
        for (size_t i = 0; i < n; ++i)
        {
            int prediction = learner->predict(this->X[i]);
            if (prediction == y[i])
            {
                weights[i] *= exp(-alpha);
            }
            else
            {
                weights[i] *= exp(alpha);
            }
        }
        // normalize the weights
        double sum_weights = 0;
        for (size_t i = 0; i < n; ++i)
        {
            sum_weights += weights[i];
        }
        for (size_t i = 0; i < n; ++i)
        {
            weights[i] /= sum_weights;
        }
    }
}

// predict using the array of all 162336 features,used in the traing phase,to find the best feature
int AdaBoost::predict(int *&X, double sl)
{
    double sum = 0;
    for (size_t i = 0; i < this->learners.size(); ++i)
    {
        sum += this->alphas[i] * (this->learners[i]->predict(X) + sl);
        // if (debug)
        //     cout << X[this->learners[i]->feature_index] << " " << this->learners[i]->threshold << " " << this->learners[i]->polarity << " " << this->learners[i]->predict(X) << endl;
    }
    // cout << endl;
    return sum >= 0.0 ? 1 : -1;
}

// predict using the 2D image,used in the testing phase
int AdaBoost::predict(int **&X, int size, double sl)
{
    double sum = 0;
    for (size_t i = 0; i < this->learners.size(); ++i)
    {
        sum += this->alphas[i] * (this->learners[i]->predict(X, size) + sl);
    }
    return sum >= 0.0 ? 1 : -1;
}

void AdaBoost::save(const string file)
{
    ofstream out(file, ios::binary);

    if (!out)
    {
        cerr << "Could not open file for writing: " << file << endl;
        return;
    }

    // Save the size of learners
    size_t learners_size = learners.size();
    out.write(reinterpret_cast<const char *>(&learners_size), sizeof(learners_size));

    // Save the alphas
    size_t alphas_size = alphas.size();
    out.write(reinterpret_cast<const char *>(&alphas_size), sizeof(alphas_size));
    out.write(reinterpret_cast<const char *>(alphas.data()), alphas_size * sizeof(double));

    // Save each learner
    for (Learner *learner : learners)
    {
        out.write(reinterpret_cast<const char *>(&learner->threshold), sizeof(learner->threshold));
        out.write(reinterpret_cast<const char *>(&learner->polarity), sizeof(learner->polarity));
        out.write(reinterpret_cast<const char *>(&learner->error), sizeof(learner->error));
        out.write(reinterpret_cast<const char *>(&learner->margin), sizeof(learner->margin));
        out.write(reinterpret_cast<const char *>(&learner->feature_index), sizeof(learner->feature_index));
    }

    out.close();
}

void AdaBoost::load(const string file)
{
    ifstream in(file, ios::binary);
    if (!in)
    {
        cerr << "Could not open file for reading: " << file << endl;
        return;
    }

    // Load the size of learners
    size_t learners_size;
    in.read(reinterpret_cast<char *>(&learners_size), sizeof(learners_size));

    // Load the alphas
    size_t alphas_size;
    in.read(reinterpret_cast<char *>(&alphas_size), sizeof(alphas_size));
    alphas.resize(alphas_size);
    in.read(reinterpret_cast<char *>(alphas.data()), alphas_size * sizeof(double));

    // Load each learner
    for (size_t i = 0; i < learners_size; ++i)
    {
        double threshold;
        int polarity;
        double error;
        double margin;
        int feature_index;

        in.read(reinterpret_cast<char *>(&threshold), sizeof(threshold));
        in.read(reinterpret_cast<char *>(&polarity), sizeof(polarity));
        in.read(reinterpret_cast<char *>(&error), sizeof(error));
        in.read(reinterpret_cast<char *>(&margin), sizeof(margin));
        in.read(reinterpret_cast<char *>(&feature_index), sizeof(feature_index));

        Learner *learner = new Learner(threshold, polarity, error, margin, feature_index);
        learners.push_back(learner);
    }

    in.close();
}

void AdaBoost::saveAsText(const string &file)
{
    ofstream out(file);
    if (!out)
    {
        cerr << "Could not open file for writing: " << file << endl;
        return;
    }

    // Save the size of learners
    out << learners.size() << endl;
    for (double alpha : alphas)
    {
        out << alpha << endl;
    }

    // Save each learner
    for (Learner *learner : learners)
    {
        out << learner->threshold << " "
            << learner->polarity << " "
            << learner->error << " "
            << learner->margin << " "
            << learner->feature_index << endl;
    }

    out.close();
}

void AdaBoost::loadFromText(const string &file)
{
    ifstream in(file);
    if (!in)
    {
        cerr << "Could not open file for reading: " << file << endl;
        return;
    }

    // Load the size of learners
    size_t learners_size;
    in >> learners_size;

    // Load the alphas
    size_t alphas_size = learners_size;
    alphas.resize(alphas_size);
    for (size_t i = 0; i < alphas_size; ++i)
    {
        in >> alphas[i];
    }

    // Load each learner
    for (size_t i = 0; i < learners_size; ++i)
    {
        double threshold;
        int polarity;
        double error;
        double margin;
        int feature_index;

        in >> threshold >> polarity >> error >> margin >> feature_index;

        Learner *learner = new Learner(threshold, polarity, error, margin, feature_index);
        learners.push_back(learner);
    }

    in.close();
}
