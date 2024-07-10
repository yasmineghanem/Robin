import cv2
import numpy as np
import pyautogui

# Initialize the video capture from the URL
HTTP = 'https://'
IP_ADDRESS = '192.168.1.141'
URL =  HTTP + IP_ADDRESS + ':4343/video'

cap = cv2.VideoCapture(URL)
    
# Load Haar Cascades for face, eye, and mouth detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variable to store the previous gray frame for optical flow
old_gray = None
# Variable to store the previous nose position
p0 = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        mid_x,mid_y=x+w/2,y+h/2
        # draw green cirlcle at the mid posiiton of the face
        cv2.circle(frame, (int(mid_x), int(mid_y)), 3, (0, 255, 0), -1)
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect mouth within the face ROI
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
            break  # Assuming the first detected mouth is the correct one

        # Detect nose within the face ROI
        # noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # if len(noses) > 0:
        #     (nx, ny, nw, nh) = noses[0]
        #     nose_center = (nx + nw // 2, ny + nh // 2)
        #     cv2.circle(roi_color, nose_center, 5, (255, 255, 0), -1)

        #     if old_gray is None:
        #         old_gray = gray.copy()
        #         p0 = np.array([nose_center], dtype=np.float32)
        #     else:
        #         p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0.reshape(-1, 1, 2), None, **lk_params)
        #         if st[0] == 1:
        #             new_nose_center = p1[0][0]
        #             cv2.circle(frame, (int(new_nose_center[0] + x), int(new_nose_center[1] + y)), 5, (0, 255, 0), -1)
        #             # move_mouse_based_on_nose_direction(p0[0], new_nose_center + [x, y])
        #             p0 = new_nose_center.reshape(-1, 1, 2)
        #             old_gray = gray.copy()

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
