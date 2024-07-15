import mmap
import os
import time
import ctypes
from ctypes.wintypes import HANDLE, BOOL, DWORD, LPCWSTR
import torch
import cv2
import numpy as np


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

    # Infinite loop to wait for C++ signal, read data, and signal back
    while True:
        # print("Waiting for the event to be signaled...")
        wait_result = kernel32.WaitForSingleObject(hEvent, INFINITE)
        if wait_result == WAIT_OBJECT_0:
            # print("Event signaled. Reading data from shared memory...")

            # Read data from shared memory
            shm.seek(0)
            image_data = shm.read(size)

            # Convert the shared memory data to a numpy array and reshape it
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((96, 96))
            # show image
            cv2.imshow('Image', image)
            # Normalize the image
            image = image.astype(np.float32) / 255.0

            # Convert the image to a Torch Tensor
            tensor_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output = model.forward(tensor_image).squeeze().numpy()

            # Plot the image with keypoints
            # plot_image_with_keypoints(image, output)
            # write in the shared memory
            # shm.seek(0)
            # shm.write(output.tobytes())
            # Convert output (x, y coordinates) to bytes
            keypoints = output.astype(np.uint8).flatten()
            keypoints_bytes = keypoints.tobytes()

            # Write the output bytes back to shared memory
            shm.seek(0)
            shm.write(keypoints_bytes)

            # Signal C++ to continue
            if not kernel32.SetEvent(hPythonEvent):
                raise ctypes.WinError(ctypes.get_last_error())

            # print("Signaled C++. Waiting for the next event...")

    # Clean up
    shm.close()

process()
