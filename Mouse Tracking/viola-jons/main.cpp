#pragma GCC optimization("Ofast")
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "AdaBoost.h"
#include "utils.h"
#include "FaceDetector.h"
#include <map>
#include <unordered_map>
#include "const.h"
#include <fstream>
using namespace std;
// TODO convert all data types to arrays instead of vectors
enum mode
{
    TRAIN_ADABOOST = 1,
    TEST_ADABOOST = 2,
    TRAIN_FACE_DETECTION = 3,
    TEST_FACE_DETECTION = 4
};

void train_ADA_BOOST(const char *file, int layers = 1, int num = -1);
void test_ADA_BOOST(const char *file, int num = -1);
void train_face_detector(const char *folder, int num, double Yo = 0.0, double Yl = 0.0, double Bl = 0.0);
void test_face_detector(const char *folder, int num);

void train_ADA_BOOST(const char *file, int layers, int num)
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
        test_ADA_BOOST("model2.txt", 500);
        cout << "---------------------------------------------\n";
    }
}

void test_ADA_BOOST(const char *file, int num)
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
    for (auto &img : pos_test)
    {
        integral_image(img, II);
        X_test.push_back(compute_haar_like_features(II));
        Y_test.push_back(1);
    }

    for (auto &img : neg_test)
    {
        integral_image(img, II);
        X_test.push_back(compute_haar_like_features(II));
        Y_test.push_back(-1);
    }
    // Example usage of the classifier
    vector<int> predeictions;
    for (auto &X : X_test)
    {
        predeictions.push_back(classifier.predict(X));
    }
    auto mat = calc_acuracy_metrices(Y_test, predeictions);
    // Print the results
    cout << "Accuracy: " << mat.accuracy << endl;
    cout << "Error rate: " << mat.error_rate << endl;
    cout << "False positive rate: " << mat.false_positive_rate << endl;
    cout << "False negative rate: " << mat.false_negative_rate << endl;
}

void train_face_detector(const char *folder, int num, double Yo, double Yl, double Bl)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string train_pos_path = "imgs/face_data_24_24/trainset/faces";
    std::string train_neg_path = "imgs/face_data_24_24/trainset/non-faces";
    vector<vector<int>> X_train;
    vector<int> Y_train;
    vector<vector<vector<int>>> X_val;
    vector<int> Y_val;

    load_features(train_pos_path, train_neg_path, X_train, Y_train, num, num);
    // TODO load the validation data
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "time for loading data is : " << duration.count() << " s\n";
    FaceDetector classifier(X_train, Y_train, X_val, Y_val);

    std::string save_folder(folder);
    std::cout << ".... Training Face Detector ....\n";
    start_time = std::chrono::high_resolution_clock::now();
    classifier.train(Yo, Yl, Bl);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Classifier trained!\n";
    duration = end_time - start_time;
    std::cout << "Training time: " << duration.count() << " s\n";
    std::cout << "saving model to " << save_folder << endl;
    classifier.save(save_folder);
}

void test_face_detector(const char *folder, int num)
{

    FaceDetector classifier;
    classifier.load(folder);
    std::string test_pos_path = "imgs/face_data_24_24/testset/faces";
    std::string test_neg_path = "imgs/face_data_24_24/testset/non-faces";
    vector<vector<vector<int>>> pos_test, neg_test;
    vector<int> Y_test;
    load_gray_images(test_pos_path, pos_test, num);
    load_gray_images(test_neg_path, neg_test, num);

    vector<vector<int>> II;
    vector<int> predeictions;
    for (auto &X : pos_test)
    {
        integral_image(X, II);
        predeictions.push_back(classifier.predict(II));
        Y_test.push_back(1);
    }
    for (auto &X : neg_test)
    {
        integral_image(X, II);
        predeictions.push_back(classifier.predict(II));
        Y_test.push_back(-1);
    }
    auto mat = calc_acuracy_metrices(Y_test, predeictions);
    // Print the results
    cout << "Accuracy: " << mat.accuracy << endl;
    cout << "Error rate: " << mat.error_rate << endl;
    cout << "False positive rate: " << mat.false_positive_rate << endl;
    cout << "False negative rate: " << mat.false_negative_rate << endl;
}

int main()
{
    // freopen("log.txt", "w", stdout);
    mode current_mode = TRAIN_FACE_DETECTION;
    if (current_mode == TRAIN_ADABOOST)
    {
        train_ADA_BOOST("model2.txt", 10, 1000);
        return 0;
    }
    else if (current_mode == TEST_ADABOOST)
    {
        test_ADA_BOOST("model2.txt", 1000);
        return 0;
    }
    else if (current_mode == TRAIN_FACE_DETECTION)
    {
        train_face_detector("face1", 1000, 0.5, 0.5, 0.5);
        // cout << "here" << endl;
        test_face_detector("face1", 500);
        return 0;
    }
    else if (current_mode == TEST_FACE_DETECTION)
    {
        test_face_detector("face1", 500);
        return 0;
    }
    return 0;
}
