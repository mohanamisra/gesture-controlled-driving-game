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


model = tf.keras.models.load_model('gesture_recogniser.keras')
y_pred = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
y_pred = np.argmax(y_pred, axis=1).tolist()
print(accuracy_score(ytrue, y_pred))
