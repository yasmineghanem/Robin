#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "AdaBoost.h"
#include "utils.h"
// namespace fs = std::filesystem;
using namespace std;
namespace fs = std::filesystem;
std::vector<std::string> get_files(const std::string &path, int num = -1)
{
    std::vector<std::string> files;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")
        {
            files.push_back(entry.path().string());
            if (num != -1 && files.size() >= num)
            {
                break;
            }
        }
    }
    return files;
}

vector<vector<vector<double>>> load_gray_images(const std::string &path, int num)
{
    vector<vector<vector<double>>> images;
    vector<string> files = get_files(path, num);

    for (const auto &file : files)
    {
        int w, h, channels;
        unsigned char *img = stbi_load(file.c_str(), &w, &h, &channels, 0);
        if (img)
        {
            std::vector<double> grayscale_image(w * h);
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
            vector<vector<double>> imageVec(h, vector<double>(w));

            // Convert to grayscale and copy data from 1D array to 2D vector
            for (int i = 0; i < h; ++i)
            {
                for (int j = 0; j < w; ++j)
                {
                    int gray_index = (i * w + j) * channels;
                    imageVec[i][j] = grayscale_image[gray_index];
                }
            }
            images.push_back(imageVec);
        }
    }
    return images;
}

int main()
{
    std::string train_pos_path = "imgs/face_data_24_24/trainset/faces";
    std::string train_neg_path = "imgs/face_data_24_24/trainset/non-faces";
    std::string test_pos_path = "imgs/face_data_24_24/testset/faces";
    std::string test_neg_path = "imgs/face_data_24_24/testset/non-faces";

    int num = 1000;
    int width = 24, height = 24;

    auto pos_train = load_gray_images(train_pos_path, num);
    auto neg_train = load_gray_images(train_neg_path, num);
    auto pos_test = load_gray_images(test_pos_path, num);
    auto neg_test = load_gray_images(test_neg_path, num);

    vector<vector<double>> X_train, X_test;
    vector<int> Y_train, Y_test;
    vector<vector<double>> II;
    for (const auto &img : pos_train)
    {
        integral_image(img, II);
        X_train.push_back(compute_haar_like_features(img, II));
        Y_train.push_back(1);
    }

    for (const auto &img : neg_train)
    {
        integral_image(img, II);
        X_train.push_back(compute_haar_like_features(img, II));
        Y_train.push_back(-1);
    }

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

    AdaBoost classifier(X_train, Y_train);
    std::cout << "Training classifier 1 layer...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    classifier.train(1);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Classifier trained!\n";
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Training time: " << duration.count() << " s\n";
    std::string file = "model.txt";
    cout << "saving model to file" << endl;
    classifier.saveAsText(file);
    // Example usage of the classifier
    vector<int> predeictions;
    for (auto &X : X_test)
    {
        predeictions.push_back(classifier.predict(X_test[0]));
    }
    int index = 0;
    // for (auto X : predeictions)
    //     cout << "h(x) = " << X << " , y = " << Y_test[index++] << " \n";
    // std::cout << std::endl;

    return 0;
}
