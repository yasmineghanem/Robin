#ifndef LANDMARK_H
#define LANDMARK_H

#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <vector>
#include <utility>
#include <windows.h>
#include <iostream>

#define NOSE_LANDMARK 0
#define LEFT_POINT_MOUSE 1
#define TOP_POINT_MOUSE 2
#define BOTTOM_POINT_MOUSE 3
#define RIGHT_POINT_MOUSE 4
class Landmark
{
private:
    dlib::shape_predictor sp;
    const std::size_t size = 10 * 1000;
    unsigned char *pBuf = nullptr;
    HANDLE hMapFile = NULL, hEvent = NULL, hPythonEvent = NULL;

public:
    Landmark();
    // load model from file
    bool load(const std::string &model_path);
    // extract landmark from face image using ML model
    std::vector<std::pair<int, int>> extract_land_mark(cv::Mat &face_image);
    // extract landmark from face image using DL model
    std::vector<std::pair<int, int>> extract_land_mark_deep(cv::Mat &face_image);
    ~Landmark();
};

#endif // LANDMARK_H
