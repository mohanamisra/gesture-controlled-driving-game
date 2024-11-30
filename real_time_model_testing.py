import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
import pygame
import random

pygame.init()

WIDTH, HEIGHT = 1200, 800
score = 0
font = pygame.font.Font(None, 74)
game_over_font = pygame.font.Font(None, 100)
game_duration = 60000
start_time = pygame.time.get_ticks()


bg_image = pygame.image.load('./assets/background.png')
car_image = pygame.image.load('./assets/car.png')
car_image = pygame.transform.scale(car_image, (80, 140))
coin_image = pygame.image.load('./assets/coin.png')
coin_image = pygame.transform.scale(coin_image, (60, 60))
bg_image = pygame.transform.scale(bg_image, (WIDTH, HEIGHT))
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Control Game")


def display_game_over():
    game_over_text = game_over_font.render("GAME OVER", True, RED)
    final_score_text = font.render(f"Final Score: {score}", True, WHITE)
    game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    score_rect = final_score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(game_over_text, game_over_rect)
    screen.blit(final_score_text, score_rect)
    pygame.display.update()
    pygame.time.delay(5000)


def generate_coins():
    coins = []
    for _ in range(3):
        coin_rect = coin_image.get_rect()
        coin_rect.x = random.randint(100, WIDTH - 100 - coin_rect.width)
        coin_rect.y = random.randint(50, HEIGHT // 2)
        coins.append(coin_rect)
    return coins


coins = generate_coins()

def display_time(elapsed_time):
    remaining_time = max(0, 60 - elapsed_time // 1000)
    time_text = font.render(f"Time: {remaining_time}", True, WHITE)
    time_rect = time_text.get_rect(center=(WIDTH - 150, 30))
    screen.blit(time_text, time_rect)
def display_score():
    score_text = font.render(f"Score: {score}", True, WHITE)
    text_rect = score_text.get_rect(center=(WIDTH // 2, 30))
    screen.blit(score_text, text_rect)

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
rect_speed = 20
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

running = True

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while running and cap.isOpened():
        elapsed_time = pygame.time.get_ticks() - start_time
        if elapsed_time >= game_duration:
            running = False
            break

        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (300, 300), interpolation=cv.INTER_AREA)
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
        sequence = sequence[:10]

        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow('Gesture Control Game', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if len(sequence) == 10:
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
                coins = generate_coins()

        rect.x += rect_velocity_x

        if rect.x < 0:
            rect.x = 0
        elif rect.x + rect.width > WIDTH:
            rect.x = WIDTH - rect.width

        for coin in coins[:]:
            if rect.colliderect(coin):
                score += 10
                coins.remove(coin)
        screen.fill(BLACK)

        screen.blit(bg_image, (0, 0))
        screen.blit(car_image, (rect.x, rect.y))
        for coin in coins:
            screen.blit(coin_image, (coin.x, coin.y))
        display_time(elapsed_time)
        display_score()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                exit()

        clock.tick(24)

screen.fill(BLACK)
display_game_over()

cap.release()
cv.destroyAllWindows()
pygame.quit()
