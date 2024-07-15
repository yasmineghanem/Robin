import torch
import cv2
import numpy as np

# Load the model
model_path = "C:/TempDesktop/fourth_year/GP/Robin/MouseTracking/landmark/keypoints_model_traced.pth"
model = torch.jit.load(model_path)
model.eval()

# Read and preprocess the image
image_path = "Untitled.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (96, 96))
imgae_color=cv2.imread(image_path)
# Normalize the image
image = image.astype(np.float32) / 255.0

# Convert the image to a Torch Tensor
tensor_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

def plot_image_with_keypoints(image_path, keypoints):
    # Read the image in grayscale
    image = cv2.imread(image_path)
    # Resize the image to 96x96
    image = cv2.resize(image, (96, 96))

    keypoints = keypoints.reshape(-1, 2)
    for (x, y) in keypoints:
        print(x, y)
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imshow('Image with Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Perform inference
with torch.no_grad():
    output = model.forward(tensor_image).squeeze().numpy()
plot_image_with_keypoints(image_path, output)
