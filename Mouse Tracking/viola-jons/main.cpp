#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "AdaBoost.h"
#include "utils.h"
#include <map>
#include <unordered_map>
using namespace std;

enum mode
{
    TRAIN = 1,
    TEST = 2
};
void train(const char *file, int layers = 1, int num = -1);
void test(const char *file, int num = -1);

void train(const char *file, int layers, int num)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string train_pos_path = "imgs/face_data_24_24/trainset/faces";
    std::string train_neg_path = "imgs/face_data_24_24/trainset/non-faces";
    vector<vector<int>> X_train;
    vector<int> Y_train;

    load_features(train_pos_path, train_neg_path, X_train, Y_train, num, num);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "time for loading data is : " << duration.count() << " s\n";
    AdaBoost classifier(X_train, Y_train);
    std::string save_file(file);
    for (int i = 0; i < layers; i++)
    {
        // std::cout << "Training classifier " << 1 << " layers...\n";
        std::cout << "Training classifier layer " << i + 1 << " \n";
        start_time = std::chrono::high_resolution_clock::now();
        classifier.train(1);
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier trained!\n";
        duration = end_time - start_time;
        std::cout << "Training time: " << duration.count() << " s\n";
        std::cout << "saving model to " << save_file << endl;
        classifier.saveAsText(save_file);
        test("model2.txt", 400);
        cout << "---------------------------------------------\n";
    }
}

void calc_acuracy_metrices(vector<int> &Y_test, vector<int> &predeictions)
{
    // Initialize counters
    int true_positive = 0, true_negative = 0;
    int false_positive = 0, false_negative = 0;

    // Calculate the counts for TP, TN, FP, and FN
    for (size_t i = 0; i < Y_test.size(); ++i)
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
    double accuracy = static_cast<double>(true_positive + true_negative) / Y_test.size();
    double error_rate = static_cast<double>(false_positive + false_negative) / Y_test.size();
    int total_positives = true_positive + false_negative;
    int total_negatives = true_negative + false_positive;
    double false_positive_rate = total_negatives > 0 ? static_cast<double>(false_positive) / total_negatives : 0;
    double false_negative_rate = total_positives > 0 ? static_cast<double>(false_negative) / total_positives : 0;

    // Print the results
    cout << "Accuracy: " << accuracy << endl;
    cout << "Error rate: " << error_rate << endl;
    cout << "False positive rate: " << false_positive_rate << endl;
    cout << "False negative rate: " << false_negative_rate << endl;
}

void test(const char *file, int num)
{

    AdaBoost classifier;
    classifier.loadFromText(file);
    std::string test_pos_path = "imgs/face_data_24_24/testset/faces";
    std::string test_neg_path = "imgs/face_data_24_24/testset/non-faces";
    vector<vector<vector<int>>> pos_test, neg_test;
    load_gray_images(test_pos_path, pos_test, num);
    load_gray_images(test_neg_path, neg_test, num);
    vector<vector<int>> X_test;
    vector<int> Y_test;
    vector<vector<int>> II;
    for (const auto &img : pos_test)
    {
        integral_image(img, II);
        X_test.push_back(compute_haar_like_features(img, II));
        Y_test.push_back(1);
    }

    for (const auto &img : neg_test)
    {
        integral_image(img, II);
        X_test.push_back(compute_haar_like_features(img, II));
        Y_test.push_back(-1);
    }
    // Example usage of the classifier
    vector<int> predeictions;
    for (auto &X : X_test)
    {
        predeictions.push_back(classifier.predict(X));
    }
    calc_acuracy_metrices(Y_test, predeictions);
}

int main()
{
    mode current_mode = TRAIN;
    // int mode;
    // cout << "Enter 1 for training and 2 for testing : ";
    // cin >> mode;
    // if (mode == 1)
    // {
    //     current_mode = TRAIN;
    // }
    // else if (mode == 2)
    // {
    //     current_mode = TEST;
    // }
    // else
    // {
    //     cout << "Invalid mode\n";
    //     return 0;
    // }
    if (current_mode == TRAIN)
    {
        train("model2.txt", 1, 1000);
        return 0;
    }
    else if (current_mode == TEST)
    {
        test("model2.txt", 400);
        return 0;
    }
    return 0;
}
