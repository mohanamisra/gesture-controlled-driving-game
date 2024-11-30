import os
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from collections import Counter
import pygame

# Initialize Pygame
pygame.init()

# Pygame window setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Control Game")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Load the pre-trained model
model = tf.keras.models.load_model('gesture_recogniser.keras')
actions = np.array(['go', 'stop', 'left', 'right', 'offence'])

# Initialize Pygame clock
clock = pygame.time.Clock()

# Define the rectangle
rectangle_width, rectangle_height = 50, 30
rect_x = WIDTH // 2 - rectangle_width // 2
rect_y = HEIGHT - rectangle_height - 10
rect_speed = 5
rect_velocity_x = 0  # Horizontal velocity
is_moving = False

# Initialize mediapipe drawing and hands modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def process_landmarks(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
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

# Initialize webcam capture
cap = cv.VideoCapture(0)

sequence = []
prediction_buffer = []

# Ensure webcam is working
if not cap.isOpened():
    print("We completely support your privacy requirements, but need to access your webcam for this app to work.")
    exit()

def draw_dashed_line(x, y1, y2, dash_length=20):
    """Draws a dashed line vertically from y1 to y2 at position x"""
    for y in range(y1, y2, dash_length * 2):
        pygame.draw.line(screen, WHITE, (x, y), (x, y + dash_length), 2)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
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

        keypoints = get_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:40]

        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow('Gesture Control Game', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if len(sequence) == 40:
            # Get the prediction for this frame
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_action = actions[np.argmax(res)]
            prediction_buffer.append(predicted_action)

        if len(prediction_buffer) >= 3:
            most_common_action, _ = Counter(prediction_buffer).most_common(1)[0]
            print(f"Predicted Action: {most_common_action}")
            prediction_buffer = []

            if most_common_action == 'go':
                is_moving = True
            elif most_common_action == 'stop':
                is_moving = False
                rect_velocity_x = 0
            elif most_common_action == 'left':
                rect_velocity_x = -rect_speed
            elif most_common_action == 'right':
                rect_velocity_x = rect_speed

        if is_moving:
            rect_y -= rect_speed
            if rect_y < -rectangle_height:
                rect_y = HEIGHT

        rect_x += rect_velocity_x

        if rect_x < 0:
            rect_x = 0
        elif rect_x + rectangle_width > WIDTH:
            rect_x = WIDTH - rectangle_width

        screen.fill(BLACK)

        pygame.draw.rect(screen, RED, pygame.Rect(rect_x, rect_y, rectangle_width, rectangle_height))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                exit()

        clock.tick(24)

cap.release()
cv.destroyAllWindows()
