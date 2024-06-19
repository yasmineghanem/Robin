from eye import *
from head import *
import threading

HTTP = 'https://'
IP_ADDRESS = '192.168.1.86'
URL =  HTTP + IP_ADDRESS + ':4343/video'

# eye_controller_object = eye_controller(URL)    
# frame_thread = threading.Thread(target=eye_controller_object.process_frames)
# frame_thread.start()


head = head_controller(URL)
frame_thread = threading.Thread(target=head.process)
frame_thread.start()


# from deepface import DeepFace
# img_path = "img1.jpg"
# analysis = DeepFace.analyze(img_path, actions=['emotion'])
# smile_probability = analysis['emotion']['happy']
# print(f'Smile Probability: {smile_probability}')


# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# # Load the image
# img_path = "img2.jpg"
# image = cv2.imread(img_path)
# height, width, _ = image.shape
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Process the image and find face landmarks
# results = face_mesh.process(rgb_image)

# if results.multi_face_landmarks:
#     for face_landmarks in results.multi_face_landmarks:
#         # Extract mouth landmarks (indices for MediaPipe are different from dlib)
#         mouth_landmarks = [face_landmarks.landmark[i] for i in [61, 291, 0, 17, 78, 308, 13, 14, 87, 317]]
#         mouth_coords = [(int(landmark.x * width), int(landmark.y * height)) for landmark in mouth_landmarks]
#         # Calculate distances to infer smile
#         left_mouth_corner = np.array(mouth_coords[0])
#         right_mouth_corner = np.array(mouth_coords[1])
#         top_lip_center = np.array(mouth_coords[2])
#         bottom_lip_center = np.array(mouth_coords[3])
#         mouth_width = np.linalg.norm(left_mouth_corner - right_mouth_corner)
#         lip_height = np.linalg.norm(top_lip_center - bottom_lip_center)
#         # A simple heuristic: if the mouth width is significantly larger than the lip height, it's likely a smile
#         smile_ratio = mouth_width / lip_height
#         is_smiling = smile_ratio > 2.0  # This threshold may need tuning
#         print(smile_ratio)
#         # Draw the mouth landmarks
#         for coord in mouth_coords:
#             cv2.circle(image, coord, 2, (0, 255, 0), -1)
#         # Display smile detection result
#         text = "Smiling" if is_smiling else "Not Smiling"
#         cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# # Display the image with landmarks and smile detection result
# cv2.imshow('Smile Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
