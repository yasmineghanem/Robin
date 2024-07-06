#pragma GCC optimization("Ofast")
#include <iostream>
// #include <vector>
#include <string>
#include <chrono>
#include "AdaBoost.h"
#include "utils.h"
#include "FaceDetector.h"
#include <map>
#include <unordered_map>
#include "const.h"
#include <fstream>
#include <tuple>
#include <thread>
#include <mutex>
#include <future>

using namespace std;
enum mode
{
    TRAIN_ADABOOST = 1,
    TEST_ADABOOST = 2,
    TRAIN_FACE_DETECTION = 3,
    TEST_FACE_DETECTION = 4
};

feature *features_info = nullptr;

// void train_ADA_BOOST(const char *file, int layers = 1, int num = -1);
// void test_ADA_BOOST(const char *file, int num = -1);
void train_face_detector(const char *folder, int num, double Yo = 0.0, double Yl = 0.0, double Bl = 0.0);

void test_face_detector(const char *folder, int num);

// void train_ADA_BOOST(const char *file, int layers, int num)
// {
//     auto start_time = std::chrono::high_resolution_clock::now();
//     std::string train_pos_path = "imgs/face_data_24_24/trainset/faces";
//     std::string train_neg_path = "imgs/face_data_24_24/trainset/non-faces";
//     vector<vector<int>> X_train;
//     vector<int> Y_train;
//     load_features(train_pos_path, train_neg_path, X_train, Y_train, num, num);
//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = end_time - start_time;
//     std::cout << "time for loading data is : " << duration.count() << " s\n";
//     AdaBoost classifier(X_train, Y_train);
//     std::string save_file(file);
//     for (int i = 0; i < layers; i++)
//     {
//         // std::cout << "Training classifier " << 1 << " layers...\n";
//         std::cout << "Training classifier layer " << i + 1 << " \n";
//         start_time = std::chrono::high_resolution_clock::now();
//         classifier.train(1);
//         end_time = std::chrono::high_resolution_clock::now();
//         std::cout << "Classifier trained!\n";
//         duration = end_time - start_time;
//         std::cout << "Training time: " << duration.count() << " s\n";
//         std::cout << "saving model to " << save_file << endl;
//         classifier.saveAsText(save_file);
//         test_ADA_BOOST("model2.txt", 500);
//         cout << "---------------------------------------------\n";
//     }
// }

// void test_ADA_BOOST(const char *file, int num)
// {
//     AdaBoost classifier;
//     classifier.loadFromText(file);
//     std::string test_pos_path = "imgs/face_data_24_24/testset/faces";
//     std::string test_neg_path = "imgs/face_data_24_24/testset/non-faces";
//     vector<vector<vector<int>>> pos_test, neg_test;
//     load_gray_images(test_pos_path, pos_test, num);
//     load_gray_images(test_neg_path, neg_test, num);
//     vector<vector<int>> X_test;
//     vector<int> Y_test;
//     vector<vector<int>> II;
//     for (auto &img : pos_test)
//     {
//         integral_image(img, II);
//         X_test.push_back(compute_haar_like_features(II));
//         Y_test.push_back(1);
//     }
//     for (auto &img : neg_test)
//     {
//         integral_image(img, II);
//         X_test.push_back(compute_haar_like_features(II));
//         Y_test.push_back(-1);
//     }
//     // Example usage of the classifier
//     vector<int> predeictions;
//     for (auto &X : X_test)
//     {
//         predeictions.push_back(classifier.predict(X));
//     }
//     auto mat = calc_acuracy_metrices(Y_test, predeictions);
//     // Print the results
//     cout << "Accuracy: " << mat.accuracy << endl;
//     cout << "Error rate: " << mat.error_rate << endl;
//     cout << "False positive rate: " << mat.false_positive_rate << endl;
//     cout << "False negative rate: " << mat.false_negative_rate << endl;
// }

void train_face_detector(const char *folder, int num, double Yo, double Yl, double Bl)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string train_pos_path = "imgs/face_data_24_24/trainset/faces";
    std::string train_neg_path = "imgs/face_data_24_24/trainset/non-faces";
    int **X_train;
    int *Y_train;
    int ***X_val;
    int *Y_val;

    std::string test_pos_path = "imgs/face_data_24_24/testset/faces";
    std::string test_neg_path = "imgs/face_data_24_24/testset/non-faces";
    int ***pos_test;
    int ***neg_test;

    pair<int, int> trian_dim = load_features(train_pos_path, train_neg_path, X_train, Y_train, num, num);
    int pos_count = load_gray_images(test_pos_path, pos_test, -1);
    int neg_count = load_gray_images(test_neg_path, neg_test, -1);
    X_val = new int **[pos_count + neg_count];
    Y_val = new int[pos_count + neg_count];
    for (int i = 0; i < pos_count; i++)
    {
        X_val[i] = pos_test[i];
        integral_image(X_val[i], X_val[i], 24, 24);
        Y_val[i] = 1;
    }
    for (int i = 0; i < neg_count; i++)
    {
        X_val[i + pos_count] = neg_test[i];
        integral_image(X_val[i + pos_count], X_val[i + pos_count], 24, 24);
        Y_val[i + pos_count] = -1;
    }
    tuple<int, int, int> val_dim(pos_count + neg_count, 24, 24);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "time for loading data is : " << duration.count() << " s\n";
    std::string save_folder(folder);
    FaceDetector classifier(X_train, Y_train, X_val, Y_val, trian_dim, val_dim, save_folder);

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
    int ***pos_test;
    int ***neg_test;
    int pos_count = load_gray_images(test_pos_path, pos_test, num);
    int neg_count = load_gray_images(test_neg_path, neg_test, num);

    int **II = new int *[24];
    for (int i = 0; i < 24; i++)
    {
        II[i] = new int[24];
    }
    int *Y_test = new int[pos_count + neg_count];
    int *predeictions = new int[pos_count + neg_count];

    for (int i = 0; i < pos_count; i++)
    {
        integral_image(pos_test[i], II, 24, 24);
        predeictions[i] = classifier.predict(II, 24);
        Y_test[i] = 1;
    }
    for (int i = 0; i < neg_count; i++)
    {
        integral_image(neg_test[i], II, 24, 24);
        predeictions[i + pos_count] = classifier.predict(II, 24);
        Y_test[i + pos_count] = -1;
    }
    auto mat = calc_acuracy_metrices(Y_test, predeictions, pos_count + neg_count);
    // Print the results
    cout << "Accuracy: " << mat.accuracy << endl;
    cout << "Error rate: " << mat.error_rate << endl;
    cout << "False positive rate: " << mat.false_positive_rate << endl;
    cout << "False negative rate: " << mat.false_negative_rate << endl;
    delete[] pos_test;
    delete[] neg_test;
    delete[] Y_test;
    delete[] predeictions;
}

void test_face_detector_threads(const char *folder, int num)
{
    FaceDetector classifier;
    classifier.load(folder);
    std::string test_pos_path = "imgs/face_data_24_24/testset/faces";
    std::string test_neg_path = "imgs/face_data_24_24/testset/non-faces";
    int ***pos_test;
    int ***neg_test;
    int pos_count = load_gray_images(test_pos_path, pos_test, num);
    int neg_count = load_gray_images(test_neg_path, neg_test, num);
    int *Y_test = new int[pos_count + neg_count];
    int *predictions = new int[pos_count + neg_count];
    // Function to process a single image
    auto process_image = [&](int ***images, int *predictions, int *Y_test, int start, int end, int lable_start, int label)
    {
        int **local_II = new int *[24];
        for (int i = 0; i < 24; i++)
        {
            local_II[i] = new int[24];
        }
        for (int i = start; i < end; ++i)
        {
            integral_image(images[i], local_II, 24, 24);
            predictions[lable_start + i] = classifier.predict(local_II, 24);
            Y_test[lable_start + i] = label;
        }
        for (int i = 0; i < 24; i++)
        {
            delete[] local_II[i];
        }
        delete[] local_II;
    };

    // Split the work among available threads
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 2; // Fallback to 2 threads if hardware_concurrency returns 0

    std::vector<std::future<void>> futures;

    // Process positive images in parallel
    int pos_chunk_size = (pos_count + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i)
    {
        int start = i * pos_chunk_size;
        int end = std::min(start + pos_chunk_size, pos_count);
        if (start < end)
        {
            futures.push_back(std::async(std::launch::async, process_image, pos_test, predictions, Y_test, start, end, 0, 1));
        }
    }

    // Process negative images in parallel
    int neg_chunk_size = (neg_count + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i)
    {
        int start = i * neg_chunk_size;
        int end = std::min(start + neg_chunk_size, neg_count);
        if (start < end)
        {
            futures.push_back(std::async(std::launch::async, process_image, neg_test, predictions, Y_test, start, end, pos_count, -1));
        }
    }
    // Wait for all threads to complete
    for (auto &f : futures)
    {
        f.get();
    }

    auto mat = calc_acuracy_metrices(Y_test, predictions, pos_count + neg_count);
    // Print the results
    std::cout << "Accuracy: " << mat.accuracy << std::endl;
    std::cout << "Error rate: " << mat.error_rate << std::endl;
    std::cout << "False positive rate: " << mat.false_positive_rate << std::endl;
    std::cout << "False negative rate: " << mat.false_negative_rate << std::endl;

    for (int i = 0; i < pos_count; ++i)
    {
        for (int j = 0; j < 24; ++j)
        {
            delete[] pos_test[i][j];
        }
        delete[] pos_test[i];
    }
    delete[] pos_test;

    for (int i = 0; i < neg_count; ++i)
    {
        for (int j = 0; j < 24; ++j)
        {
            delete[] neg_test[i][j];
        }
        delete[] neg_test[i];
    }
    delete[] neg_test;

    delete[] Y_test;
    delete[] predictions;
}

int main()
{

    fill_features_info();
    // cout << (int)features_info[0].i << endl;
    // cout << (int)features_info[123000].j << endl;
    // cout << (int)features_info[100500].w << endl;
    // cout << (int)features_info[150500].h << endl;
    // return 0;
    // freopen("log.txt", "w", stdout);
    mode current_mode = TRAIN_FACE_DETECTION;
    if (current_mode == TRAIN_ADABOOST)
    {
        // train_ADA_BOOST("model2.txt", 10, 1000);
        return 0;
    }
    else if (current_mode == TEST_ADABOOST)
    {
        // test_ADA_BOOST("model2.txt", 1000);
        return 0;
    }
    else if (current_mode == TRAIN_FACE_DETECTION)
    {
        train_face_detector("face1", 100, 0.2, 0.5, 0.2);
        test_face_detector_threads("face1", -1);
        return 0;
    }
    else if (current_mode == TEST_FACE_DETECTION)
    {
        auto start = std::chrono::high_resolution_clock::now();
        test_face_detector_threads("face1", -1);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Testing time: " << duration.count() << " s\n";
        return 0;
    }

    return 0;
}
