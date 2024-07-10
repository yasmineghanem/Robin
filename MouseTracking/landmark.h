#ifndef LANDMARK_H
#define LANDMARK_H

#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <vector>
#include <utility>

#define NOSE_LANDMARK 0
#define LEFT_POINT_MOUSE 1
#define TOP_POINT_MOUSE 2
#define BOTTOM_POINT_MOUSE 3
#define RIGHT_POINT_MOUSE 4
class Landmark
{
public:
    bool load(const std::string &model_path);
    std::vector<std::pair<int, int>> extract_land_mark(cv::Mat &face_image);

private:
    dlib::shape_predictor sp;
};

#endif // LANDMARK_H
