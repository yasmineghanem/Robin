#include <opencv2/opencv.hpp>
#include "landmark.h"
#include "mousecontroller.h"
#include <iostream>
#include <algorithm>
#include "./viola-jons/FaceDetector.h"
#include "./viola-jons/const.h"
#include "./viola-jons/const.h"
#include "./viola-jons/AdaBoost.h"
#include "./viola-jons/utils.h"
#include "./viola-jons/FaceDetector.h"

using namespace std;
using namespace cv;
const string HTTP = "https://";
const string IP_ADDRESS = "192.168.1.2";
const string URL = HTTP + IP_ADDRESS + ":4343/video";
feature *features_info = nullptr;

int **matTo2DArray(const cv::Mat &mat)
{
    int **array = new int *[mat.rows];
    for (int i = 0; i < mat.rows; ++i)
    {
        array[i] = new int[mat.cols];
        for (int j = 0; j < mat.cols; ++j)
        {
            cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
            // Average the RGB values
            array[i][j] = static_cast<int>(pixel[0] / 3.0 + pixel[1] / 3.0 + pixel[2] / 3.0);
        }
    }
    return array;
}

int ***matTo3DArray(const cv::Mat &mat)
{
    int ***array = new int **[mat.rows];
    for (int i = 0; i < mat.rows; ++i)
    {
        array[i] = new int *[mat.cols];
        for (int j = 0; j < mat.cols; ++j)
        {
            array[i][j] = new int[3];
            cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
            array[i][j][0] = pixel[0];
            array[i][j][1] = pixel[1];
            array[i][j][2] = pixel[2];
        }
    }
    return array;
}

int main()
{
    fill_features_info();

    MouseController mouse_controller;
    // int x = 500, y = 500;
    // while (true)
    // {
    //     mouse_controller.move_mouse(x, y);
    //     x += 10;
    //     y += 10;

    //     Sleep(1000);
    //     mouse_controller.left_click();
    //     Sleep(30);
    //     mouse_controller.left_click();
    //     Sleep(30);
    //     mouse_controller.left_click();
    //     _sleep(1000);
    //     mouse_controller.right_click();
    // }

    // Load the cascade and landmark model
    FaceDetector face_cascade;
    face_cascade.load("face7");

    // cv::CascadeClassifier classifier;
    Landmark landmark_extractor;
    if (!landmark_extractor.load("shape_predictor_68_face_landmarks.dat"))
    {
        return -1;
    }
    // if (!face_cascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml")))
    // {
    //     std::cerr << "Error loading face cascade\n";
    //     return -1;
    // }
    // cv::VideoCapture cap(URL);
    // if (!cap.isOpened())
    // {
    //     std::cerr << "Error opening video stream" << std::endl;
    //     return -1;
    // }
    while (true)
    {
        cv::Mat fliped;
        // cap >> fliped;
        //
        // cv::Mat frame;
        // cv::flip(fliped, frame, 1);
        cv::Mat frame = cv::imread("img7.jpg", IMREAD_COLOR);

        if (frame.empty())
        {
            break;
        }
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        // Detect faces
        // std::vector<cv::Rect> faces;
        // Convert cv::Mat to int** and int***
        int **grayArray = matTo2DArray(frame);
        int ***colorArray = matTo3DArray(frame);
        int M = frame.rows;
        int N = frame.cols;
        // load_image("img1.jpg", colorArray, grayArray, M, N);
        // Get the dimensions

        // Call the process function
        double c = 1.5; // Example parameter
        auto faces = face_cascade.process(grayArray, colorArray, M, N, c);
        sort(faces.begin(), faces.end(), [](window *a, window *b)
             { 
                int y1=a->y;
                int y2=b->y;
                int x1=a->x;
                int x2=b->x;
                int w1=a->w;
                int w2=b->w;
                if (y1 != y2)
                    return y1 < y2;
                if (x1 != x2)
                    return x1 < x2;
                return w1 < w2; });
        int maxi = -1;
        for (int i = 0; i < faces.size(); i++)
        {
            maxi = max(maxi, faces[i]->w);
        }
        for (size_t i = 0; i < std::min<size_t>(100000, faces.size()); i++)
        {
            if (faces[i]->w < maxi / 2)
                continue;
            cv::Rect rect(faces[i]->y, faces[i]->x, faces[i]->w, faces[i]->w);
            // std::cout << "Face " << i + 1 << ": x=" << faces[i]->y << ", y=" << faces[i]->x
            //           << ", width=" << faces[i]->w << ", height=" << rect.height << std::endl;

            cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
            cv::Mat face = frame(rect);

            std::vector<std::pair<int, int>> landmarks = landmark_extractor.extract_land_mark(face);
            // cout << landmarks.size() << endl;
            for (auto &landmark : landmarks)
            {
                // find the abslute position of the landmark
                landmark.first += faces[i]->y;
                landmark.second += faces[i]->x;
            }
            for (auto &landmark : landmarks)
            {
                // cout << landmark.first << " " << landmark.second << endl;
                cv::circle(frame, cv::Point(landmark.first, landmark.second), 2, cv::Scalar(0, 255, 0), -1);
            }
            // mouse_controller.control(landmarks);
            // std::string windowName = "Face " + std::to_string(i + 1);
            // cv::imshow("img", face);
            // cv::waitKey(0);
            // cv::destroyAllWindows();
        }

        // Display the flipped frame
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        cv::imshow("Frame", frame);
        cv::waitKey(0);
        if (cv::waitKey(10) == 'q')
        {
            break;
        }
        break;
    }

    // cap.release();
    cv::destroyAllWindows();
    return 0;
}