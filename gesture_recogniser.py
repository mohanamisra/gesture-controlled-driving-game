import numpy as np
import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("We completely support your privacy requirements, but need to access your webcam for this app to work.")
    exit()

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
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
        cv.imshow('Webcam', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

sign = []

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # hand_landmarks.landmark GIVES THE ITERABLE HAND LANDMARK OBJECT, WHERE THE LENGTH IS 21
        # print(len(hand_landmarks.landmark))
        for landmark in hand_landmarks.landmark:
            sign.append(np.array([landmark.x, landmark.y, landmark.z]))

sign = np.array(sign).flatten()
print(sign.shape)

cap.release()
cv.destroyAllWindows()
