#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>

// Function to plot image with keypoints
void plot_image_with_keypoints(const std::string &image_path, const std::vector<float> &keypoints)
{
    // Read the image in color
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    // Resize the image to 96x96
    cv::resize(image, image, cv::Size(96, 96));

    for (size_t i = 0; i < keypoints.size(); i += 2)
    {
        int x = static_cast<int>(keypoints[i]);
        int y = static_cast<int>(keypoints[i + 1]);
        std::cout << x << " " << y << std::endl;
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("Image with Keypoints", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main()
{
    // Load the model
    std::string model_path = "C:/TempDesktop/fourth_year/GP/Robin/MouseTracking/landmark/keypoints_model_traced.pth";
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.eval();

    // Read and preprocess the image
    std::string image_path = "Untitled.png";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(96, 96));

    cv::Mat image_color = cv::imread(image_path); // Original colored image for display

    // Normalize the image
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    // Convert the image to a Torch Tensor
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, 1, 96, 96}, torch::kFloat32);

    // Perform inference
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);
    at::Tensor output = model.forward(inputs).toTensor().squeeze();
    std::vector<float> keypoints(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());

    // Plot the image with keypoints
    plot_image_with_keypoints(image_path, keypoints);

    return 0;
}
