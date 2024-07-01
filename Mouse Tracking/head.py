from mouse_utilities import *
from eye import *
import cv2
import mediapipe as mp
import numpy as np

class head_controller():
    def __init__(self,URL):
        self.is_left = False
        self.is_right = False
        self.is_up = False
        self.is_down = False
        self.move=False
        self.mp_face_mesh = mp.solutions.face_mesh
        self.URL = URL 
        self.last_point=None
        self.counter = 0
        self.threshold = 10
    def distance_equ(self,distane):
        return distane**1.7
    def detect_horizontal_movement(self,point1,point2,frame):
        x1=point1.x * frame.shape[1]
        x2=point2.x * frame.shape[1]
        diffrence=abs(x1-x2)
        if diffrence>self.threshold:
            self.move=True
            if x1>x2:
                return directions.RIGHT,self.distance_equ(diffrence)
            else:
                return directions.LEFT,self.distance_equ(diffrence)
        else:
            self.move=False
            return directions.STRAIGHT,0
    def detect_vertical_movement(self,point1,point2,frame):
        y1=point1.y * frame.shape[0]
        y2=point2.y * frame.shape[0]
        diffrence=abs(y1-y2)
        if diffrence>self.threshold:
            if y1>y2:
                return directions.UP,self.distance_equ(diffrence)
            else:
                return directions.DOWN,self.distance_equ(diffrence)
        else:
            self.move=False
            return directions.STRAIGHT,0
    def is_smile(self,face_landmarks,frame):
        height, width, _ = frame.shape
        # Extract mouth landmarks (indices for MediaPipe are different from dlib)
        mouth_landmarks = [face_landmarks.landmark[i] for i in [61, 291, 0, 17, 78, 308, 13, 14, 87, 317]]
        mouth_coords = [(int(landmark.x * width), int(landmark.y * height)) for landmark in mouth_landmarks]
        # Calculate distances to infer smile
        left_mouth_corner = np.array(mouth_coords[0])
        right_mouth_corner = np.array(mouth_coords[1])
        top_lip_center = np.array(mouth_coords[2])
        bottom_lip_center = np.array(mouth_coords[3])
        mouth_width = np.linalg.norm(left_mouth_corner - right_mouth_corner)
        lip_height = np.linalg.norm(top_lip_center - bottom_lip_center)
        # A simple heuristic: if the mouth width is significantly larger than the lip height, it's likely a smile
        smile_ratio = mouth_width / lip_height
        is_smiling = smile_ratio > 4.0  # This threshold may need tuning
        # print(smile_ratio)
        return is_smiling
    def process(self):
        eye = eye_controller(self.URL)
        self.cap = cv2.VideoCapture(self.URL)
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as face_mesh:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    break
                result = face_mesh.process(frame)
                if result.multi_face_landmarks is None :
                    continue
                for face_landmarks in result.multi_face_landmarks:
                    nose = face_landmarks.landmark[4]
                    if self.last_point is not None:
                        horizontal_movement,distance_horizontal=self.detect_horizontal_movement(self.last_point,nose,frame)
                        vertical_movement,distance_vertical=self.detect_vertical_movement(self.last_point,nose,frame)
                        eye.left_blink(face_landmarks,frame)
                        eye.right_blikc(face_landmarks,frame)
                        if horizontal_movement == directions.LEFT:
                            self.is_left=True
                            self.is_right=False
                        elif horizontal_movement == directions.RIGHT:
                            self.is_right=True
                            self.is_left=False
                        else:
                            nose.x = self.last_point.x
                            self.is_left=False
                            self.is_right=False
                        # ///////////////////////
                        if vertical_movement == directions.UP:
                            self.is_up=True
                            self.is_down=False
                        elif vertical_movement == directions.DOWN:
                            self.is_down=True
                            self.is_up=False
                        else:
                            nose.y=self.last_point.y
                            self.is_up=False
                            self.is_down=False
                    self.last_point=nose
                    if self.is_smile(face_landmarks,frame):
                        if self.is_left:
                            move_mouse(directions.LEFT,distance_horizontal)
                        if self.is_right:
                            move_mouse(directions.RIGHT,distance_horizontal)
                        if self.is_up:
                            move_mouse(directions.UP,distance_vertical)
                        if self.is_down:
                            move_mouse(directions.DOWN,distance_vertical)
                        if(self.move==False):
                            move_mouse(directions.STRAIGHT,0)
                        if(eye.is_left_blink):
                            mouse_click(CLICK.LEFT_CLICK)
                        elif(eye.is_right_blink):
                            mouse_click(CLICK.RIGHT_CLICK)
                cv2.imshow('frame', cv2.flip(frame,1))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
