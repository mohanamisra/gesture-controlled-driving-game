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


sign = []
def process_landmarks(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # hand_landmarks.landmark GIVES THE ITERABLE HAND LANDMARK OBJECT, WHERE THE LENGTH IS 21
            # print(len(hand_landmarks.landmark))
            temp_curr_hand = []
            for landmark in hand_landmarks.landmark:
                temp_curr_hand.append([landmark.x, landmark.y, landmark.z])
            return np.array(temp_curr_hand)
    else:
        return np.zeros(21 * 3)


def get_keypoints(results):
    temp_arr = process_landmarks(results)
    np_sign = temp_arr.flatten()
    return np_sign


path = os.path.join('data')
actions = np.array(['go', 'stop', 'left', 'right', 'offence'])
no_of_videos = 20
video_frame_len = 40

# TEST VARS
# actions = np.array(['go', 'stop'])
# no_of_videos = 3
# video_frame_len = 5

for action in actions:
    for video in range(no_of_videos):
        try:
            os.makedirs(os.path.join(path, action, str(video)))
        except:
            pass


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("We completely support your privacy requirements, but need to access your webcam for this app to work.")
    exit()

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    for action in actions:
        for video in range(no_of_videos):
            for frame_num in range(video_frame_len):
                ret, frame = cap.read()
                frame = cv.flip(frame, 1)
                if not ret:
                    print("Exiting stream...")
                    break
                img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = hands.process(img)

                img.flags.writeable = True

                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=1, circle_radius=1),
                                                  mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2, circle_radius=2))

                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

                if frame_num == 0:
                    cv.putText(img, "STARTING COLLECTION", (120, 200), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv.LINE_AA)
                    cv.putText(img, f"Collecting frames for {video} for action {action}", (15, 12), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv.LINE_AA)
                    cv.waitKey(2000)
                else:
                    cv.putText(img, f"Collecting frames for {video} for action {action}", (15, 12), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv.LINE_AA)

                frame_keypoints = get_keypoints(results)
                keypoints_path = os.path.join(path, action, str(video), str(frame_num))
                np.save(keypoints_path, frame_keypoints)

                cv.imshow("Webcam", img)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break


cap.release()
cv.destroyAllWindows()
