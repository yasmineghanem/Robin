#include <torch/script.h>     // One-stop header for LibTorch
#include <opencv2/opencv.hpp> // Include OpenCV headers

int main()
{
    // Load the model
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load("C:/TempDesktop/fourth_year/GP/Robin/MouseTracking/landmark/keypoints_model.pth");
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the model: " << e.msg() << "\n";
        return -1;
    }

    // Set the model to evaluation mode
    module.eval();

    // Assuming you have an image in OpenCV format (BGR) and it's resized to 96x96
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(96, 96));

    // Normalize the image
    image.convertTo(image, CV_32FC1, 1.0 / 255.0);

    // Convert the OpenCV Mat to a Torch Tensor
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, 96, 96}).clone();
    tensor_image = tensor_image.unsqueeze(0); // Add a batch dimension

    // Perform inference
    at::Tensor output = module.forward({tensor_image}).toTensor();

    // Convert output tensor to std::vector<float>
    std::vector<float> keypoints;
    for (int i = 0; i < output.size(1); ++i)
    {
        keypoints.push_back(output[0][i].item<float>());
    }

    // Assuming output is a tensor with shape [1, 14] containing keypoints
    // Draw keypoints on the image
    for (int i = 0; i < keypoints.size(); i += 2)
    {
        int x = static_cast<int>(keypoints[i] * 96);                      // Scale keypoints back to image size
        int y = static_cast<int>(keypoints[i + 1] * 96);                  // Scale keypoints back to image size
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1); // Draw a red circle
    }

    // Show the image with keypoints
    cv::imshow("Image with Predicted Keypoints", image);
    cv::waitKey(0);

    return 0;
}
