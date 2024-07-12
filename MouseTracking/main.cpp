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
bool isVertical(Rect rect)
{
    return rect.height > rect.width * 1.5;
}

bool isHorizontal(Rect rect)
{
    return rect.width > rect.height * 1.5;
}
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

bool contains(cv::Rect rect, std::vector<cv::Rect> &rects)
{
    for (auto &r : rects)
    {
        if (rect.contains(cv::Point(r.x, r.y)) && rect.contains(cv::Point(r.x + r.width, r.y + r.height)))
        {
            return true;
        }
    }
    return false;
}

void extract_rect(cv::Mat &frame, vector<Rect> &verticalRectangles,
                  vector<Rect> &horizontalRectangles)
{

    // Convert to grayscale
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Apply edge detection
    Mat edges;
    Canny(gray, edges, 50, 150);

    // Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Filter contours to identify rectangles
    vector<Rect> rectangles;
    for (size_t i = 0; i < contours.size(); i++)
    {
        Rect rect = boundingRect(contours[i]);
        double aspectRatio = (double)rect.width / rect.height;
        // Assume rectangles with aspect ratio in range [0.5, 2] are valid
        if (aspectRatio >= 0.5 && aspectRatio <= 2)
        {
            rectangles.push_back(rect);
        }
    }

    // Separate vertical and horizontal rectangles
    for (const Rect &rect : rectangles)
    {
        if (isVertical(rect))
        {
            verticalRectangles.push_back(rect);
        }
        else if (isHorizontal(rect))
        {
            horizontalRectangles.push_back(rect);
        }
    }

    // Draw rectangles on the image
    Mat output = frame.clone();
    for (const Rect &rect : verticalRectangles)
    {
        rectangle(output, rect, Scalar(0, 255, 0), 2); // Green for vertical rectangles
    }
    for (const Rect &rect : horizontalRectangles)
    {
        rectangle(output, rect, Scalar(255, 0, 0), 2); // Blue for horizontal rectangles
    }

    // Show the result
    // namedWindow("Detected Rectangles", WINDOW_AUTOSIZE);
    // imshow("Detected Rectangles", output);
    // waitKey(0);
}

void resizeToMaxDimension(Mat &image, int maxDim)
{
    // Get the original dimensions
    int originalWidth = image.cols;
    int originalHeight = image.rows;

    // Calculate aspect ratio
    double aspectRatio = static_cast<double>(originalWidth) / originalHeight;

    // Calculate new dimensions
    int newWidth, newHeight;
    if (originalWidth > originalHeight)
    {
        newWidth = maxDim;
        newHeight = static_cast<int>(maxDim / aspectRatio);
    }
    else
    {
        newHeight = maxDim;
        newWidth = static_cast<int>(maxDim * aspectRatio);
    }

    // Resize the image
    resize(image, image, Size(newWidth, newHeight));
}

int main()
{

    const char *filename = "img2.jpg";
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
    face_cascade.load("face18_best");

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
        cv::Mat frame = cv::imread(filename, IMREAD_COLOR);

        if (frame.empty())
        {
            break;
        }
        resizeToMaxDimension(frame, 400);
        // Convert to grayscale
        // cv::Mat gray;
        // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
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
        double c = 1.1;
        auto faces = face_cascade.process(grayArray, colorArray, M, N, c);
        int maxi = -1;
        int n = 0;
        for (int i = 0; i < faces.size(); i++)
        {
            n++;
            maxi = max(maxi, faces[i]->w);
        }
        cout << n << endl;
        for (size_t i = 0; i < n; i++)
        {
            if (faces[i]->w != maxi)
                continue;
            // cout << faces[i]->x << " " << faces[i]->y << " " << faces[i]->w << endl;
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
