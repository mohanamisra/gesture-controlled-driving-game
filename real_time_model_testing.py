import os
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
import pygame

pygame.init()

WIDTH, HEIGHT = 1200, 800
bg_image = pygame.image.load('./assets/background.png')
car_image = pygame.image.load('./assets/car.png')
car_image = pygame.transform.scale(car_image, (80, 140))
bg_image = pygame.transform.scale(bg_image, (WIDTH, HEIGHT))
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Control Game")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

model = tf.keras.models.load_model('gesture_recogniser.keras')
actions = np.array(['go', 'stop', 'left', 'right', 'offence'])

clock = pygame.time.Clock()

rect = car_image.get_rect()
rectangle_width, rectangle_height = 50, 30
rect.x = WIDTH // 2 - rect.width // 2
rect.y = HEIGHT - rect.height - 10
rect_speed = 5
rect_velocity_x = 0
is_moving = False

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


cap = cv.VideoCapture(0)

sequence = []
prediction_buffer = []

if not cap.isOpened():
    print("We completely support your privacy requirements, but need to access your webcam for this app to work.")
    exit()

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
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_action = actions[np.argmax(res)]

            if predicted_action == 'go':
                is_moving = True
            elif predicted_action == 'stop':
                is_moving = False
                rect_velocity_x = 0
            elif predicted_action == 'left':
                rect_velocity_x = -rect_speed
            elif predicted_action == 'right':
                rect_velocity_x = rect_speed
            else:
                is_moving = False
                rect_velocity_x = 0

        if is_moving:
            rect.y -= rect_speed
            if rect.y < -rect.height:
                rect.y = HEIGHT

        rect.x += rect_velocity_x

        if rect.x < 0:
            rect.x = 0
        elif rect.x + rect.width > WIDTH:
            rect.x = WIDTH - rect.width

        screen.fill(BLACK)

        screen.blit(bg_image, (0, 0))
        screen.blit(car_image, (rect.x, rect.y))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                exit()

        clock.tick(24)

cap.release()
cv.destroyAllWindows()
