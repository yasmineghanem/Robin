#include <opencv2/opencv.hpp>
#include "landmark.h"
#include "mousecontroller.h"
#include <iostream>
#include <algorithm>
using namespace std;
using namespace cv;
const string HTTP = "https://";
const string IP_ADDRESS = "192.168.1.2";
const string URL = HTTP + IP_ADDRESS + ":4343/video";

int main()
{
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
    cv::CascadeClassifier face_cascade;
    Landmark landmark_extractor;
    if (!landmark_extractor.load("shape_predictor_68_face_landmarks.dat"))
    {
        return -1;
    }
    if (!face_cascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml")))
    {
        std::cerr << "Error loading face cascade\n";
        return -1;
    }
    cv::VideoCapture cap(URL);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }
    while (true)
    {
        cv::Mat fliped;
        cap >> fliped;
        cv::Mat frame;
        cv::flip(fliped, frame, 1);
        // cv::Mat frame = cv::imread("img1.jpg", IMREAD_COLOR);
        if (frame.empty())
        {
            break;
        }
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 10);

        // Initialize Landmark extractor

        // Process each detected face
        for (size_t i = 0; i < std::min<size_t>(1, faces.size()); i++)
        {
            cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);
            cv::Mat face = frame(faces[i]); // Use the color frame to draw the circles
            std::vector<std::pair<int, int>> landmarks = landmark_extractor.extract_land_mark(face);
            for (auto &landmark : landmarks)
            {
                // find the abslute position of the landmark
                landmark.first += faces[i].x;
                landmark.second += faces[i].y;
            }
            for (auto &landmark : landmarks)
            {
                cout << landmark.first << " " << landmark.second << endl;
                cv::circle(frame, cv::Point(landmark.first, landmark.second), 2, cv::Scalar(0, 255, 0), -1);
            }
            mouse_controller.control(landmarks);
        }

        // Display the flipped frame
        cv::imshow("Frame", frame);
        if (cv::waitKey(10) == 'q')
        {
            break;
        }
    }

    // cap.release();
    cv::destroyAllWindows();
    return 0;
}