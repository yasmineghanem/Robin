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

template <typename T>
void alocate2DArray(T **&array, int rows, int cols)
{
    array = new T *[rows];
    for (int i = 0; i < rows; i++)
    {
        array[i] = new T[cols];
    }
}

template <typename T>
void allocate3DArray(T ***&array, int rows, int cols, int depth)
{
    array = new T **[rows];
    for (int i = 0; i < rows; i++)
    {
        array[i] = new T *[cols];
        for (int j = 0; j < cols; j++)
        {
            array[i][j] = new T[depth];
        }
    }
}

void matTo2DArray(int **array, const cv::Mat &mat)
{
    // int **array = new int *[mat.rows];
    for (int i = 0; i < mat.rows; ++i)
    {
        // array[i] = new int[mat.cols];
        for (int j = 0; j < mat.cols; ++j)
        {
            cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
            // Average the RGB values
            array[i][j] = static_cast<int>(pixel[0] / 3.0 + pixel[1] / 3.0 + pixel[2] / 3.0);
        }
    }
}

void matTo3DArray(int ***array, const cv::Mat &mat)
{
    // int ***array = new int **[mat.rows];
    for (int i = 0; i < mat.rows; ++i)
    {
        // array[i] = new int *[mat.cols];
        for (int j = 0; j < mat.cols; ++j)
        {
            // array[i][j] = new int[3];
            cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
            array[i][j][0] = pixel[0];
            array[i][j][1] = pixel[1];
            array[i][j][2] = pixel[2];
        }
    }
    // return array;
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

FaceDetector face_cascade_strong;
FaceDetector face_cascade_weak;

int main()
{

    const char *filename = "imgs/img6.jpg";
    fill_features_info();
    MouseController mouse_controller;

    // Load the cascade and landmark model
    face_cascade_strong.load("face18_best");
    face_cascade_weak.load("face18_weak");
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
    cv::VideoCapture cap(URL);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }
    int processed = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    // read one frame to initialize the frame size
    cv::Mat fliped;
    cap >> fliped;
    cv::Mat frame;
    resizeToMaxDimension(fliped, 400);
    // initialize the arrays to carry the data
    int **grayArray;
    int ***colorArray;
    alocate2DArray(grayArray, fliped.rows, fliped.cols);
    allocate3DArray(colorArray, fliped.rows, fliped.cols, 3);

    // allocate memory used in the process operation by waak face detector
    int **skin_denisty;
    int **II;
    long long **IIsq;
    alocate2DArray(skin_denisty, fliped.rows, fliped.cols);
    alocate2DArray(II, fliped.rows, fliped.cols);
    alocate2DArray(IIsq, fliped.rows, fliped.cols);

    // allocate memory used in the process operation by strong face detector
    int **skin_denisty_strong;
    int **II_strong;
    long long **IIsq_strong;
    alocate2DArray(skin_denisty_strong, fliped.rows, fliped.cols);
    alocate2DArray(II_strong, fliped.rows, fliped.cols);
    alocate2DArray(IIsq_strong, fliped.rows, fliped.cols);

    face_cascade_strong.stride = 2;
    face_cascade_weak.stride = 2;
    thread strong_thread(&FaceDetector::infinite_prcess, &face_cascade_strong);
    thread weak_thread(&FaceDetector::infinite_prcess, &face_cascade_weak);

    // store the last top left corner of the face
    int last_x = 70;
    int last_y = 130;
    int last_size = 180;
    bool first = false;

    // process the video stream
    vector<window *> faces;
    while (true)
    {
        bool strong = false;
        // Capture frame-by-frame
        cap >> fliped;
        // If the frame is empty, break immediately
        if (fliped.empty())
        {
            break;
        }
        // Flip the frame horizontally because the camera is mirrored
        cv::flip(fliped, frame, 1);

        // Resize the frame to a maximum dimension of 400 to fast the process
        resizeToMaxDimension(frame, 400);

        // Convert to grayscale
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Convert cv::Mat to int** and int***
        matTo2DArray(grayArray, frame);
        matTo3DArray(colorArray, frame);
        int M = frame.rows;
        int N = frame.cols;
        double c = 1.1;
        // run the first frame with the string thread , to initialize last_x and last_y
        // if (!first)
        // {
        //     face_cascade_strong.set_in(II_strong, IIsq_strong, skin_denisty_strong, grayArray, colorArray, M, N, c, 0, 0, M, N, 1000, 4);
        //     face_cascade_strong.start_flag = true;
        //     first = true;
        //     auto start = std::chrono::high_resolution_clock::now();
        //     while (!face_cascade_strong.end_flag)
        //     {
        //         this_thread::sleep_for(std::chrono::milliseconds(10));
        //     }
        //     auto end = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        //     cout << "Time taken for the first strong frame is : " << duration.count() << " milliseconds" << endl;
        //     faces = face_cascade_strong.res.windows;
        //     // relase the end flag
        //     face_cascade_strong.end_flag = false;
        //     // store if the cuurect result is strong or weak
        //     // because if it is strong so i will need to calc the last_x and last_y
        //     strong = true;
        // }
        // else
        // {
        // if the current frame is not the first frame , so i will run both the strong and weak face detector in parallel

        // check first if the weak is running on frame or not , also check if it has ready result or not
        if (face_cascade_weak.start_flag == 0 && face_cascade_weak.end_flag == 0)
        {
            face_cascade_weak.set_in(II, IIsq, skin_denisty, grayArray, colorArray, M, N, c, last_x - 50, last_y - 50, last_x + 50, last_y + 50, last_size + 50, 1);
            face_cascade_weak.start_flag = true;
        }
        if (face_cascade_strong.start_flag == 0 && face_cascade_weak.end_flag == 0)
        {
            face_cascade_strong.set_in(II_strong, IIsq_strong, skin_denisty_strong, grayArray, colorArray, M, N, c, 0, 0, M, N, 1000, 4);
            face_cascade_strong.start_flag = true;
        }
        // here the state is that each one of the weak and strong face detector is running on a frame , or at least has result ready to read

        // if there is no ready result , then wait for it
        while (!face_cascade_weak.end_flag && !face_cascade_strong.end_flag)
        {
            this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // check wich one give us result , give priority to the strong face detector
        if (face_cascade_strong.end_flag)
        {
            faces = face_cascade_strong.res.windows;
            face_cascade_strong.end_flag = false;
            strong = true;
        }
        else
        {
            faces = face_cascade_weak.res.windows;
            face_cascade_weak.end_flag = false;
        }
        // }

        // Detect faces
        // auto faces = face_cascade_strong.process_part(II, IIsq, skin_denisty, grayArray, colorArray, M, N, c, 0, 0, 1000, 1000, 2000, 3);

        // sort the faces by the y coordinate , to take the last maximum face
        // sort(faces.begin(), faces.end(), [](window *a, window *b)
        //      { return a->y < b->w; });
        // loop over te faces to find the maximum face and its index
        int maxi = -1;
        int index = -1;
        int n = 0;
        for (int i = 0; i < faces.size(); i++)
        {
            n++;
            if (faces[i]->w > maxi)
            {
                maxi = faces[i]->w;
                index = i;
            }
        }
        if (index == -1)
            continue;

        // if the current result is strong , then i will store the last_x and last_y
        if (strong)
        {
            // cout << "this is strong result" << endl;
            last_x = faces[index]->x;
            last_y = faces[index]->y;
            last_size = faces[index]->w;
            // cout << "strong result : " << last_x << " " << last_y << " " << last_size << endl;
        }
        // else
        // {
        //      cout << "this is weak result" << endl;
        // }
        // for (size_t i = 0; i < n; i++)
        // {
        //     if (faces[i]->w != maxi)
        //         continue;
        // cout << faces[index]->x << " " << faces[index]->y << " " << faces[index]->w << endl;
        cv::Rect rect(faces[index]->y, faces[index]->x, faces[index]->w, faces[index]->w);
        cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
        cv::Mat face = frame(rect);
        std::vector<std::pair<int, int>> landmarks = landmark_extractor.extract_land_mark(face);
        for (auto &landmark : landmarks)
        {
            // find the abslute position of the landmark
            landmark.first += faces[index]->y;
            landmark.second += faces[index]->x;
            cv::circle(frame, cv::Point(landmark.first, landmark.second), 2, cv::Scalar(0, 255, 0), -1);
        }
        // mouse_controller.control(landmarks);
        // }

        // Display the frame with the landmarks
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        cv::imwrite("output.jpg", frame);
        cv::imshow("Frame", frame);
        // cv::waitKey(0);
        if (cv::waitKey(10) == 'q')
        {
            break;
        }
        if (processed == 10)
            start = std::chrono::high_resolution_clock::now();
        processed++;
        if (processed == 40)
        {
            end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            cout << "number of processed frames : " << processed - 10 << endl;
            cout << "Time taken: " << duration.count() << " milliseconds" << endl;
            // face_cascade_strong.set_in(II, IIsq, skin_denisty, grayArray, colorArray, M, N, c, 0, 0, M, N, 1000, 4);
            // face_cascade_strong.start_flag = true;
            // break;
        }
        for (auto &face : faces)
        {
            delete face;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
