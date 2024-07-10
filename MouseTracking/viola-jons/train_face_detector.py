import cv2
import numpy as np
import os
import sys
import time
from face_detector import *

train_pos_path = 'imgs/face_data/trainset/faces'
train_neg_path = 'imgs/face_data/trainset/non-faces'
test_pos_path = 'imgs/face_data/testset/faces'
test_neg_path = 'imgs/face_data/testset/non-faces'

def load_images(path,num=None,converter=cv2.COLOR_BGR2GRAY):
    images = []
    files= os.listdir(path)
    if(num):
        files = files[:num]
    for file in files:
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(path, file))
            if(converter):
                img = cv2.cvtColor(img,converter)
            images.append(img)
    return images

num = 10
pos_train = load_images(train_pos_path,num,converter=cv2.COLOR_BGR2GRAY)
neg_train = load_images(train_neg_path,num,converter=cv2.COLOR_BGR2GRAY)
pos_test = load_images(test_pos_path,num,converter=cv2.COLOR_BGR2GRAY)
neg_test = load_images(test_neg_path,num,converter=cv2.COLOR_BGR2GRAY)

X_train = []
Y_train = []
X_test = []
Y_test = []
num= 10

for img in pos_train:
    # resize img to 24x24
    img = cv2.resize(img, (24, 24))
    X_train.append(compute_haar_like_features(img,integral_image(img)))
    Y_train.append(1)
    
for img in neg_train:
    img = cv2.resize(img, (24, 24))
    X_train.append(compute_haar_like_features(img,integral_image(img)))
    Y_train.append(0)  
      
for img in pos_test:
    img = cv2.resize(img, (24, 24))
    X_test.append(compute_haar_like_features(img,integral_image(img)))
    Y_test.append(1)    
    
for img in neg_test:
    img = cv2.resize(img, (24, 24))
    X_test.append(compute_haar_like_features(img,integral_image(img)))
    Y_test.append(0)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print('Training classifier...')
start_time  = time.time()
classifier = adaboost(X_train, Y_train, 1)
end = time.time()
print('Classifier trained!')
print('Training time: ', end - start_time, 's')
classifier(X_test[0])