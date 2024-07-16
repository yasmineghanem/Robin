import mmap
import os
import time
import ctypes
from ctypes.wintypes import HANDLE, BOOL, DWORD, LPCWSTR
import torch
import cv2
import numpy as np
import mediapipe as mp


def process():

    # Define constants
    EVENT_ALL_ACCESS = 0x1F0003
    WAIT_OBJECT_0 = 0x00000000
    INFINITE = 0xFFFFFFFF

    # Load the Windows API
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    # Define the wait function
    kernel32.WaitForSingleObject.argtypes = [HANDLE, DWORD]
    kernel32.WaitForSingleObject.restype = DWORD

    # Define the open event function
    kernel32.OpenEventW.argtypes = [DWORD, BOOL, LPCWSTR]
    kernel32.OpenEventW.restype = HANDLE

    # Define the create event function
    kernel32.CreateEventW.argtypes = [ctypes.c_void_p, BOOL, BOOL, LPCWSTR]
    kernel32.CreateEventW.restype = HANDLE

    # Define the set event function
    kernel32.SetEvent.argtypes = [HANDLE]
    kernel32.SetEvent.restype = BOOL

    # Define the fixed size of the shared memory
    size = 96 * 96 

    # Infinite loop to create/open shared memory and events
    while True:
        try:
            # Connect to or create the shared memory
            shm = mmap.mmap(-1, size, tagname="Local\\MyFixedSizeSharedMemory")
            break
        except OSError:
            print("Waiting to create or open shared memory...")
            time.sleep(1)

    # Open or create the event for notification from C++
    while True:
        hEvent = kernel32.OpenEventW(EVENT_ALL_ACCESS, False, "Local\\MyEvent")
        if hEvent:
            break
        else:
            hEvent = kernel32.CreateEventW(None, False, False, "Local\\MyEvent")
            if hEvent:
                break
        print("Waiting to create or open MyEvent...")
        time.sleep(1)

    # Open or create the event to signal C++
    while True:
        hPythonEvent = kernel32.OpenEventW(EVENT_ALL_ACCESS, False, "Local\\PythonEvent")
        if hPythonEvent:
            break
        else:
            hPythonEvent = kernel32.CreateEventW(None, False, False, "Local\\PythonEvent")
            if hPythonEvent:
                break
        print("Waiting to create or open PythonEvent...")
        time.sleep(1)

    # Load the model
    model_path = "C:/TempDesktop/fourth_year/GP/Robin/MouseTracking/landmark/keypoints_model_traced.pth"
    model = torch.jit.load(model_path)
    model.eval()

    def plot_image_with_keypoints(image, keypoints):
        # Resize the image to 96x96
        image = cv2.resize(image, (96, 96))

        keypoints = keypoints.reshape(-1, 2)
        for (x, y) in keypoints:
            print(x, y)
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

        cv2.imshow('Image with Keypoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Initialize MediaPipe FaceMesh model
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    # Infinite loop to wait for C++ signal, read data, and signal back
    while True:
        wait_result = kernel32.WaitForSingleObject(hEvent, INFINITE)
        if wait_result == WAIT_OBJECT_0:
            # Read data from shared memory
            shm.seek(0)
            image_data = shm.read(size)
            # Convert the shared memory data to a numpy array and reshape it
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((96, 96))
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Run MediaPipe FaceMesh on the image
            results = face_mesh.process(image)
            keypoints = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # add the nose keypoint
                    land_mark_number =4 
                    keypoints.extend([face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]])
                    
                    # outer left eye
                    land_mark_number =33
                    keypoints.extend([face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]])
                    # print(face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0])
                    # inner left eye
                    land_mark_number =173
                    keypoints.extend([face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]])
                    # top most left eye
                    land_mark_number = 160
                    p2x,p2y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    land_mark_number = 158
                    p3x,p3y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    keypoints.extend([(p2x+p3x)/2, (p2y+p3y)/2])
                    # bottom most left eye
                    land_mark_number = 155
                    p2x,p2y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    land_mark_number = 163
                    p3x,p3y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    keypoints.extend([(p2x+p3x)/2, (p2y+p3y)/2])

                    
                    # outer right eye
                    land_mark_number =398
                    keypoints.extend([face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]])
                    # print(face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0])
                    # inner right eye
                    land_mark_number =466
                    keypoints.extend([face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]])
                    # top most right eye
                    land_mark_number = 385
                    p2x,p2y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    land_mark_number = 387
                    p3x,p3y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    keypoints.extend([(p2x+p3x)/2, (p2y+p3y)/2])
                    # bottom most right eye
                    land_mark_number = 373
                    p2x,p2y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    land_mark_number = 380
                    p3x,p3y=face_landmarks.landmark[land_mark_number].x * image.shape[1], face_landmarks.landmark[land_mark_number].y* image.shape[0]
                    keypoints.extend([(p2x+p3x)/2, (p2y+p3y)/2])

                    break
                    for landmark in face_landmarks.landmark:
                        # Extract landmark coordinates
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        # keypoints.extend([x, y])
                        # Draw or process landmarks as needed
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Example: Draw a dot on each landmark
                    break
            # Display or save processed image
            # cv2.imshow("Image with Landmarks", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # Write the output bytes back to shared memory
            keypoints_array = np.array(keypoints, dtype=np.uint8)
            shm.seek(0)
            shm.write(keypoints_array.tobytes())
            # Signal C++ to continue
            if not kernel32.SetEvent(hPythonEvent):
                raise ctypes.WinError(ctypes.get_last_error())


    # Clean up
    shm.close()

process()
import cv2
import mediapipe as mp

def extract_landmarks(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at '{image_path}'")
        return
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 96x96
    image = cv2.resize(image, (96, 96))
    # Initialize MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Process image
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Extract landmarks if detection is successful
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # Extract landmark coordinates
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                # Draw or process landmarks as needed
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Example: Draw a dot on each landmark

    # Display or save processed image
    cv2.imshow("Image with Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "Untitled2.png"
extract_landmarks(image_path)

