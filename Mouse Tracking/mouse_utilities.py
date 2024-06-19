import cv2
from enum import Enum
import pyautogui
class directions(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    STRAIGHT = 0
class CLICK(Enum):
    LEFT_CLICK = 1
    RIGHT_CLICK = 2
    
def move_mouse(direction,distance=5):
    if(direction == directions.STRAIGHT):
        return
    current_x, current_y = pyautogui.position()
    if direction == directions.LEFT:
        current_x -= distance
    elif direction == directions.RIGHT:
        current_x += distance
    elif direction == directions.UP:
        current_y -= distance
    elif direction == directions.DOWN:
        current_y += distance
    else:
        print('mouse is in the same position')
    pyautogui.moveTo(int(current_x), int(current_y))
def mouse_click( click_type):
    if click_type == CLICK.LEFT_CLICK:
        pyautogui.click(button='left')
    elif click_type == CLICK.RIGHT_CLICK:
        pyautogui.click(button='right')
    else:
        print("Unknown Click Type")
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml') 

def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
  
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return frame 
