import cv2
import mediapipe as mp
import pyautogui
import ipywidgets as widgets
from IPython.display import display, clear_output
from mouse_utilities import *


class eye_controller:
    def __init__(self,URL):
        self.is_left = False
        self.is_right = False
        self.is_up = False
        self.is_down = False
        self.move=False
        
        self.is_left_blink = False
        self.is_right_blink = False
        self.EYE_AR_THRESH = 0.25    #threshold for eye aspect ratio , if the ratio is less than this value then the eye is closed
        self.EYE_AR_CONSEC_FRAMES_left = 7  #number of frames to say there is blink
        self.EYE_AR_CONSEC_FRAMES_right = 15 #number of frames to say there is blink
        self.COUNTER_left = 0 #count 7 franes before say there is blink
        self.COUNTER_right = 0 #count 7 franes before say there is blink

        self.horeizontal_threshold = 7
        self.verticall_threshold =1 
        
        self.right_eye_mid_index= 468
        
        self.right_eye_topmost_eye = 159
        self.right_eye_downmost_eye = 145
        
        
        self.right_eye_leftmost_eye = 173
        self.right_eye_leftmost_pupil = 469
        
        self.right_eye_rightmost_eye = 33
        self.right_eye_rightmost_pupil = 471

        self.left_eye_topmost_eye = 386
        self.left_eye_downmost_eye = 374
        
        self.left_eye_mid_index = 473
        self.left_eye_leftmost = 362
        self.left_eye_rightmost = 466

        self.right_eye_boundary = [33,  133, 173, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155,133]
        self.left_eye_boundary = [362, 263, 373, 380, 381, 382, 384, 385, 386, 387, 388, 466, 390, 373, 374, 380]
        self.mp_face_mesh = mp.solutions.face_mesh
        self.URL=URL
        

    def neared_distance(self,point0,point1,point2,frame):
        x_0 = int(point0.x * frame.shape[1])
        y_0 = int(point0.y * frame.shape[0])
        
        x_1 = int(point1.x * frame.shape[1])
        y_1 = int(point1.y * frame.shape[0])
        
        x_2 = int(point2.x * frame.shape[1])
        y_2 = int(point2.y * frame.shape[0])
        
        # calc the distance between point 0 and point 1
        distance_1 = ((x_0 - x_1)**2 + (y_0 - y_1)**2)**0.5
        
        # calc the distance between point 0 and point 2
        distance_2 = ((x_0 - x_2)**2 + (y_0 - y_2)**2)**0.5
        # print(distance_1,distance_2)
        return distance_1,distance_2

    def distance(self,point0,point1,frame):
        x_0 = int(point0.x * frame.shape[1])
        y_0 = int(point0.y * frame.shape[0])
        
        x_1 = int(point1.x * frame.shape[1])
        y_1 = int(point1.y * frame.shape[0])
        
        # calc the distance between point 0 and point 1
        distance = ((x_0 - x_1)**2 + (y_0 - y_1)**2)**0.5
        return distance
    
    def get_ratio(self,mid_point,point1,point2,frame):
        total_distance = float(self.distance(point1,point2,frame))
        left_distance = float(self.distance(mid_point,point1,frame))
        return left_distance/total_distance
    
    def vertical_decision(self,mid_point, top_most, bottom_most, frame):
        distance_1,distance_2 = self.neared_distance(mid_point, top_most, bottom_most,frame)
        if abs(distance_1 - distance_2) < self.verticall_threshold:
            return directions.STRAIGHT
        elif distance_1 < distance_2:
            return directions.UP
        else:
            return directions.DOWN
        
    def horizontal_desecion(self,mid_point, left_most, right_most,frame):
        distance_1,distance_2 = self.neared_distance(mid_point, left_most, right_most,frame)
        distance_2-=5
        # print('horizontal_distances: ',distance_1,distance_2)
        if abs(distance_1 - distance_2) < self.horeizontal_threshold:
            return directions.STRAIGHT
        elif distance_1 < distance_2:
            return directions.LEFT
        else:
            return directions.RIGHT


    def horizontal_process(self,face_landmarks,frame):
        self.is_left = False
        self.is_right = False
        p1=face_landmarks.landmark[self.right_eye_rightmost_eye]
        p2=face_landmarks.landmark[self.right_eye_rightmost_pupil]
        
        p3=face_landmarks.landmark[self.right_eye_leftmost_pupil]
        p4=face_landmarks.landmark[self.right_eye_leftmost_eye]
        
        right_distance = self.distance(p1,p2,frame)
        left_distance = self.distance(p3,p4,frame)
        diffrence= abs(right_distance - left_distance)
        if( diffrence > min(right_distance,left_distance)):
            self.move=True
            if(right_distance < left_distance):
                self.is_right=True
            else :
                self.is_left=True
    
    def vertitical_process(self,face_landmarks,frame):
        self.is_up=False
        self.is_down=False
        p1 = face_landmarks.landmark[self.right_eye_mid_index]
        p1.y+=0.001
        p2= face_landmarks.landmark[self.right_eye_topmost_eye]
        p3= face_landmarks.landmark[self.right_eye_downmost_eye]
        top_distance = self.distance (p1,p2,frame)
        down_distnace = self.distance(p1,p3,frame)
        
        diffrence = abs(top_distance - down_distnace)*1.2
        if(diffrence > min(top_distance,down_distnace)):
            self.move=True
            if(top_distance < down_distnace):
                self.is_up=True
            else:
                self.is_down=True
    
    def eye_aspect_ratio(self,p1,p2,p3,p4,p5,p6,frame):
        # print(p2,p6,p3,p5,p1,p4)
        distance_1 = self.distance(p2,p6,frame)
        distance_2 = self.distance(p3,p5,frame)
        distance_3 = self.distance(p1,p4,frame)
        return (distance_2 + distance_1)/(2.0 * distance_3)
    
    def left_blink(self,face_landmarks,frame):
        self.is_left_blink = False
        p1=face_landmarks.landmark[398]
        p2=face_landmarks.landmark[385]
        p3=face_landmarks.landmark[387]
        p4=face_landmarks.landmark[466]
        p5=face_landmarks.landmark[373]
        p6=face_landmarks.landmark[380]
        ear = self.eye_aspect_ratio(p1,p2,p3,p4,p5,p6,frame)   
        if ear<self.EYE_AR_THRESH:
            self.COUNTER_left+=1
            if self.COUNTER_left>=self.EYE_AR_CONSEC_FRAMES_left:
                self.is_left_blink=True
                self.COUNTER_left=0
    
    def right_blikc(self,face_landmarks,frame):
        self.is_right_blink = False
        p1=face_landmarks.landmark[33]
        p2=face_landmarks.landmark[160]
        p3=face_landmarks.landmark[158]
        p4=face_landmarks.landmark[173]
        p5=face_landmarks.landmark[155]
        p6=face_landmarks.landmark[163]
        ear = self.eye_aspect_ratio(p1,p2,p3,p4,p5,p6,frame)      
        if ear<self.EYE_AR_THRESH:
            self.COUNTER_right+=1
            if self.COUNTER_right>=self.EYE_AR_CONSEC_FRAMES_right:
                self.is_right_blink=True
                self.COUNTER_right=0
                
        
    def process_frames(self):        
        self.cap = cv2.VideoCapture(self.URL)
        distance = 10
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
                    self.move=False
                    self.horizontal_process(face_landmarks,frame)
                    self.vertitical_process(face_landmarks,frame)
                    self.left_blink(face_landmarks,frame)
                    self.right_blikc(face_landmarks,frame)
                    if self.is_left_blink:
                        mouse_click(CLICK.LEFT_CLICK)
                    elif self.is_right_blink:
                        mouse_click(CLICK.RIGHT_CLICK)
                    if self.is_left:
                        move_mouse(directions.LEFT,distance)
                    if self.is_right:
                        move_mouse(directions.RIGHT,distance)
                    if self.is_up:
                        move_mouse(directions.UP,distance)
                    if self.is_down:
                        move_mouse(directions.DOWN,distance)
                    if(self.move==False):
                        move_mouse(directions.STRAIGHT)


                cv2.imshow('frame', cv2.flip(frame,1))
                # Add a delay of 10 milliseconds
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            
        self.cap.release()
        cv2.destroyAllWindows()