import cv2
import numpy as np

# Load the image
image_path = 'color1.png'
image = cv2.imread(image_path)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Loop over the face detections
for (x, y, w, h) in faces:
    # Define the region of interest (ROI) for the forehead
    roi_gray = gray[y:y + h//2, x:x + w]
    roi_color = image[y:y + h//2, x:x + w]
    
    # Apply GaussianBlur to the ROI to reduce noise
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # show
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Apply morphological operations to enhance the horizontal edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # show 
    cv2.drawContours(roi_color, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', roi_color)
    cv2.waitKey(0)
    # Filter contours that might correspond to eyebrows based on size and position
    for contour in contours:
        (cx, cy, cw, ch) = cv2.boundingRect(contour)
        
        # Consider only the contours that are wider than they are tall (near-horizontal)
        if ch < cw:
            cv2.rectangle(roi_color, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

    # Show the result with detected eyebrows
    cv2.imshow('Eyebrows', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
