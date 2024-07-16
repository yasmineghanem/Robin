#include "landmark.h"
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>

using namespace dlib;
using namespace std;
using namespace cv;

Landmark::Landmark()
{
    this->load("shape_predictor_68_face_landmarks.dat");
    // return;
    // Infinite loop to create/open shared memory and events
    while (true)
    {
        // Create or open the shared memory object
        hMapFile = CreateFileMapping(
            INVALID_HANDLE_VALUE,                    // Use paging file
            NULL,                                    // Default security
            PAGE_READWRITE,                          // Read/Write access
            0,                                       // Maximum object size (high-order DWORD)
            size,                                    // Maximum object size (low-order DWORD)
            TEXT("Local\\MyFixedSizeSharedMemory")); // Name of the mapping object

        if (hMapFile != NULL)
        {
            pBuf = (unsigned char *)MapViewOfFile(
                hMapFile,            // Handle to the map object
                FILE_MAP_ALL_ACCESS, // Read/Write access
                0,
                0,
                size);

            if (pBuf != NULL)
                break;
        }

        // Clean up if creation/opening failed
        if (hMapFile)
            CloseHandle(hMapFile);

        std::cerr << "Waiting to create or open shared memory..." << std::endl;
        Sleep(1000);
    }

    // Create or open the event for signaling Python that data is ready
    while (true)
    {
        hEvent = CreateEvent(
            NULL,                    // Default security attributes
            FALSE,                   // Auto-reset event
            FALSE,                   // Initial state is non-signaled
            TEXT("Local\\MyEvent")); // Name of the event

        if (hEvent != NULL)
            break;

        std::cerr << "Waiting to create or open MyEvent..." << std::endl;
        Sleep(1000);
    }

    // Create or open the event for receiving signal from Python
    while (true)
    {
        hPythonEvent = CreateEvent(
            NULL,                        // Default security attributes
            FALSE,                       // Auto-reset event
            FALSE,                       // Initial state is non-signaled
            TEXT("Local\\PythonEvent")); // Name of the event

        if (hPythonEvent != NULL)
            break;

        std::cerr << "Waiting to create or open PythonEvent..." << std::endl;
        Sleep(1000);
    }
}
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
std::vector<std::pair<int, int>> Landmark::extract_land_mark_deep(cv::Mat &face_image)
{
    // return vector with zeros
    // std::vector<std::pair<int, int>> landmarks2(7, std::make_pair(0, 0));
    // return landmarks2;
    // write the image in the shared memory
    for (int i = 0; i < face_image.rows; i++)
        for (int j = 0; j < face_image.cols; j++)
            pBuf[i * face_image.cols + j] = face_image.at<uchar>(i, j);
    // notify Python that the image is ready
    SetEvent(hEvent);
    // wait for result
    WaitForSingleObject(hPythonEvent, INFINITE);
    // read the result from the shared memory and return it
    std::vector<std::pair<int, int>> landmarks;
    for (int i = 0; i < 9; i++)
    {
        int x = (int)pBuf[i * 2];
        int y = (int)pBuf[i * 2 + 1];
        landmarks.push_back({x, y});
    }
    return landmarks;
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

Landmark::~Landmark()
{
    return;
    if (pBuf)
        UnmapViewOfFile(pBuf);
    if (hMapFile)
        CloseHandle(hMapFile);
    if (hEvent)
        CloseHandle(hEvent);
    if (hPythonEvent)
        CloseHandle(hPythonEvent);
}