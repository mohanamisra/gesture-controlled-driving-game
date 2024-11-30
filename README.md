# Gear Gesture 1.0 - A Hand-Gesture Controlled Car Driving Game 🏁🚗🚙

### So you think you can drive manual?  
   <img src="https://github.com/user-attachments/assets/61dbb349-bdb9-4f78-8081-b1badf75dcc7" width="500"/>
   <img src="https://github.com/user-attachments/assets/aecc41fd-21af-43a0-b44c-5202de1b50c3" width="500" height="300"/>


A hand-gesture-controlled top-down car game. Collect as many coins as possible within 60 seconds.

#### Controls: 
#### (Remember to use your right hand!)
1. Open Palm for "GO"  
   <img src="https://github.com/user-attachments/assets/958d30cd-3126-40a3-95f7-ce9116c76b84" width="200" />

2. Closed Fist for "STOP"  
   <img src="https://github.com/user-attachments/assets/de096287-4f89-4afb-8e3b-84634bc901c9" width="200" />

3. Thumb and Pointer out ("L-hand gesture") for "LEFT"  
   <img src="https://github.com/user-attachments/assets/51a50801-8d35-4463-aaba-f2cd95e90d93" width="200" />

4. Pointer and Middle fingers out ("V-hand gesture") for "RIGHT"  
   <img src="https://github.com/user-attachments/assets/747a70eb-a47e-4a13-9198-4af9a2edfd74" width="200" />


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
