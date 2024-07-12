#include <iostream>
#include <fstream>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <opencv2/opencv.hpp>
// Helper function to read an image file and preprocess it
std::vector<float> LoadAndPreprocessImage(const std::string &image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        throw std::runtime_error("Failed to read image from: " + image_path);
    }
    cv::resize(image, image, cv::Size(96, 96));
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    std::vector<float> input_image;
    input_image.assign((float *)image.datastart, (float *)image.dataend);
    return input_image;
}

int main()
{
    // Load the TFLite model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("model.tflite");
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
    {
        std::cerr << "Failed to build interpreter" << std::endl;
        return 1;
    }

    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return 1;
    }

    // Load and preprocess the image
    std::vector<float> input_image = LoadAndPreprocessImage("img1.jpg");

    // Set the input tensor
    float *input = interpreter->typed_input_tensor<float>(0);
    std::copy(input_image.begin(), input_image.end(), input);

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    // Get the output tensor
    float *output = interpreter->typed_output_tensor<float>(0);

    // Print the keypoints
    for (int i = 0; i < interpreter->outputs().size(); ++i)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
