# were divided into two sets containing 2451 and 446 images respectively for training and validation.
# 1000 training and 1000 validation positive example

import cv2
import os 
import numpy as np
from sklearn.model_selection import train_test_split

train_pos_path2 = "imgs/face_data_24_24/trainset/faces"
train_neg_path2 = "imgs/face_data_24_24/trainset/non-faces"
test_pos_path2 = "imgs/face_data_24_24/testset/faces"
test_neg_path2 = "imgs/face_data_24_24/testset/non-faces"

train_pos_path = "imgs/face_data_24_24_old/trainset/faces"
train_neg_path = "imgs/face_data_24_24_old/trainset/non-faces"
test_pos_path = "imgs/face_data_24_24_old/testset/faces"
test_neg_path = "imgs/face_data_24_24_old/testset/non-faces"

# laod all positive and negative images
train_pos = [cv2.imread(os.path.join(train_pos_path, file), cv2.IMREAD_GRAYSCALE) for file in os.listdir(train_pos_path)]
test_pos = [cv2.imread(os.path.join(test_pos_path, file), cv2.IMREAD_GRAYSCALE) for file in os.listdir(test_pos_path)]
train_neg = [cv2.imread(os.path.join(train_neg_path, file), cv2.IMREAD_GRAYSCALE) for file in os.listdir(train_neg_path)]
test_neg = [cv2.imread(os.path.join(test_neg_path, file), cv2.IMREAD_GRAYSCALE) for file in os.listdir(test_neg_path)]
# delete all data in folder face_data_24_24
for file in os.listdir(train_pos_path2):
    os.remove(os.path.join(train_pos_path2, file))
for file in os.listdir(train_neg_path2):
    os.remove(os.path.join(train_neg_path2, file))
for file in os.listdir(test_pos_path2):
    os.remove(os.path.join(test_pos_path2, file))
for file in os.listdir(test_neg_path2):
    os.remove(os.path.join(test_neg_path2, file))

pos_count= len(train_pos) + len(test_pos)
neg_count= len(train_neg) + len(test_neg)
print(pos_count)
print(neg_count)

# randomly devide the positive data between validation and training
all_pos = train_pos + test_pos
all_neg = train_neg + test_neg
train_pos_new, test_pos_new = train_test_split(all_pos, test_size=0.5, random_state=42)

# Save the divided positive images into respective directories
for i, img in enumerate(train_pos_new):
    cv2.imwrite(os.path.join(train_pos_path2, f"train_pos_{i}.png"), img)

for i, img in enumerate(test_pos_new):
    cv2.imwrite(os.path.join(test_pos_path2, f"test_pos_{i}.png"), img)

all_neg= train_neg + test_neg
all_neg= all_neg[:4000]
train_neg_new, test_neg_new = train_test_split(all_pos, test_size=0.25, random_state=42)

# Save the divided positive images into respective directories
for i, img in enumerate(train_neg_new):
    cv2.imwrite(os.path.join(train_neg_path2, f"train_neg_{i}.png"), img)

for i, img in enumerate(test_neg_new):
    cv2.imwrite(os.path.join(test_neg_path2, f"test_neg_{i}.png"), img)

