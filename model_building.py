import os
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import accuracy_score

path = os.path.join('data')
actions = np.array(['go', 'stop', 'left', 'right', 'offence'])
no_of_videos = 20
video_frame_len = 40


label_map = {label: num for num, label in enumerate(actions)}

# TEST FUNC (TO CHECK IF CORRECT LENGTH OF ARRAYS
# for action in actions:
#     for video in range(no_of_videos):
#         for frame in range(video_frame_len):
#             temp = np.load(os.path.join(path, action, str(video), f"{frame}.npy"))
#             print(temp.shape)

features, targets = [], []
for action in actions:
    for video in range(no_of_videos):
        window = []
        for frame_num in range(video_frame_len):
            temp = np.load(os.path.join(path, action, str(video), f"{frame_num}.npy"))
            window.append(temp)
        features.append(window)
        targets.append(label_map[action])

X = np.array(features)
y = to_categorical(targets).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

log_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(32, input_shape=(40, 63), return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(
    X_train,
    y_train,
    epochs=350,
    validation_split=0.25,
    callbacks=[tb_callback]
)

model.save('gesture_recogniser.keras')
