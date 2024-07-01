import cv2
import os 
import numpy as np
train_pos_path = "imgs/face_data/trainset/faces"
train_neg_path = "imgs/face_data/trainset/non-faces"
test_pos_path = "imgs/face_data/testset/faces"
test_neg_path = "imgs/face_data/testset/non-faces"

new_train_pos_path = "imgs/face_data_24_24/trainset/faces"
new_train_neg_path = "imgs/face_data_24_24/trainset/non-faces"
new_test_pos_path = "imgs/face_data_24_24/testset/faces"
new_test_neg_path = "imgs/face_data_24_24/testset/non-faces"

# create new folder if not exist
if not os.path.exists(new_train_pos_path):
    os.makedirs(new_train_pos_path)
if not os.path.exists(new_train_neg_path):
    os.makedirs(new_train_neg_path)
if not os.path.exists(new_test_pos_path):
    os.makedirs(new_test_pos_path)
if not os.path.exists(new_test_neg_path):
    os.makedirs(new_test_neg_path)
    
num = 1000
for file in os.listdir(train_pos_path)[:num]:
    img = cv2.imread(os.path.join(train_pos_path, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (24, 24))
    cv2.imwrite(os.path.join(new_train_pos_path, file), img)

for file in os.listdir(train_neg_path)[:num]:
    img = cv2.imread(os.path.join(train_neg_path, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (24, 24))
    cv2.imwrite(os.path.join(new_train_neg_path, file), img)

for file in os.listdir(test_pos_path)[:num]:
    img = cv2.imread(os.path.join(test_pos_path, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (24, 24))
    cv2.imwrite(os.path.join(new_test_pos_path, file), img)

for file in os.listdir(test_neg_path)[:num]:
    img = cv2.imread(os.path.join(test_neg_path, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (24, 24))
    cv2.imwrite(os.path.join(new_test_neg_path, file), img)
