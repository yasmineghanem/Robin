#include "landmark.h"
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>

using namespace dlib;
using namespace std;
using namespace cv;

bool Landmark::load(const std::string &model_path)
{
    try
    {
        dlib::deserialize(model_path) >> sp;
        return true;
    }
    catch (std::exception &e)
    {
        std::cerr << "Error loading shape predictor model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::pair<int, int>> Landmark::extract_land_mark(cv::Mat &face_image)
{
    std::vector<std::pair<int, int>> landmarks;

    // Convert OpenCV image to Dlib image
    dlib::cv_image<dlib::bgr_pixel> dlib_image(face_image);

    // Detect face landmarks
    std::vector<dlib::rectangle> dets = {dlib::rectangle(0, 0, face_image.cols, face_image.rows)};
    dlib::full_object_detection shape = sp(dlib_image, dets[0]);

    // Indices of the required landmarks
    std::vector<int> indices = {30, 48, 50, 52, 56, 58, 64};

    // Collect the required landmark points
    for (int idx : indices)
    {
        dlib::point pt = shape.part(idx);
        // cout << pt.x() << " " << pt.y() << endl;
        landmarks.emplace_back(pt.x(), pt.y());
    }
    // cout << " end \n";
    return landmarks;
}
