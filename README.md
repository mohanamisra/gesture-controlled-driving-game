# Gear Gesture 1.0 - A Hand-Gesture Controlled Car Driving Game 🏁🚗🚙

### So you think you can drive manual?

A hand-gesture-controlled top-down car game. Collect as many coins as possible within 60 seconds.

#### Controls:
1. Open Palm for "GO"
2. Closed Fist for "STOP"
3. Thumb and Pointer out ("L-hand gesture") for "LEFT"
4. Pointer and Middle fingers out ("V-hand gesture") for "RIGHT"

#### Requirements:
- OpenCV
- MediaPipe
- Tensorflow
- Pygame

#### How to Run the Game:

1. Clone/Fork this repository  
   ```
   # if you want to clone this repository...
   
   git clone git@github.com:mohanamisra/gesture-controlled-driving-game.git
   cd gesture-controlled-driving-game
   
   # otherwise click the fork button on GitHub and clone your own fork
    ```
2. Install requirements  
   ```
   pip install mediapipe
   python3 -m pip install 'tensorflow[and-cuda]'
   pip install pygame
   pip install opencv-python
   ```
3. Run `app.py` located in the root of the folder
4. <b>(Optional)</b> Add your own gestures to the game by modifying variables in the `model_building` folder located at the root
5. <b>(Optional)</b> Star this repository if you like it 😊