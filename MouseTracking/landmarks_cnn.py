import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Paths to the CSV files
train_path = '/kaggle/working/training.csv'
test_path = '/kaggle/working/test.csv'
lookid_path = '/kaggle/input/facial-keypoints-detection/IdLookupTable.csv'

# Read the CSV files
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
lookid_data = pd.read_csv(lookid_path)

# Columns for Y_train
y_columns = [
    'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y',
    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
]

# Select the Image column for X_train
X_train = train_data['Image']

# Select the specified columns for Y_train
Y_train = train_data[y_columns]

# Concatenate X_train and Y_train to discard rows with missing values
combined_data = pd.concat([X_train, Y_train], axis=1)

# Drop rows with any missing values
combined_data = combined_data.dropna()

# Split the combined data back into X_train and Y_train
X_train = combined_data['Image']
Y_train = combined_data[y_columns]

# Convert X_train to numpy array and reshape
X_train = np.array([np.fromstring(image, sep=' ').reshape(96, 96, 1) for image in X_train])
X_train = X_train.astype('float32') / 255.0  # Normalize pixel values

# Convert Y_train to numpy array
Y_train = np.array(Y_train)

# Define the model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(96, 96, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, Y_train, epochs=350, batch_size=32, validation_split=0.2)

# Save the model
# model.save('path/to/your/model')
