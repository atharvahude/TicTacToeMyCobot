# TicTacToe Game with Object Detection and Robotic Manipulation

## Author
- Atharva Jitendra Hude
- ASU ID: 1229854940
- Arizona State University

## Table of Contents
- [Introduction](#introduction)
- [Method](#method)
  - [Perception Part](#perception-part)
  - [Game Logic](#game-logic)
  - [Useful Code Snippets](#useful-code-snippets)
  - [Project Flow](#project-flow)

## Introduction

This project demonstrates the integration of OpenCV, YoloV8 object detection, and MyCobot to create a unique Tic-Tac-Toe game with object detection and robotic manipulation. The game involves a human player and a robot player using stickers of different objects, such as basketball and apple, as their symbols. The camera captures the grid, the YoloV8 algorithm locates the stickers, and the MyCobot robot arm makes the moves.

### Demo Link: [Youtube Link](https://www.youtube.com/watch?v=1AKvKo89SIc)


## Method

### Perception Part

1. **Create a Dataset:**
   - Decide the class to detect (e.g., basketball or apple).
   - Capture images from the cobot’s camera or create a synthetic dataset.
   - Annotate the dataset using the RoboFlow tool.

2. **Train the Object Detection Model:**
   - Utilize a pre-trained YOLO model and fine-tune it on the custom dataset using Google Colab.
   - Example dataset: [YoloTraining Dataset](https://universe.roboflow.com/yolotraining-2lreo/bassketball)

### Game Logic

- Define a 3x3 grid as a list of lists, with each element representing a space in the grid.
- Functions include printing the grid, checking if the grid is full, checking for a winner, getting moves from human and robot players, and playing the game.
  
### Useful Code Snippets

```python
# Turn the pump on and off
def pump(state):
    if state:
        mc.set_gripper_state(100, 0)
    else:
        mc.set_gripper_state(0, 0)

# Move the robot arm to a specific angle
def move(angle):
    mc.send_angles(angle, 50)
```

### Project Flow

1. Take the user’s input to define the region of interest (ROI) from the camera.
2. Draw a rectangle on the screen using the cv2.mouse_callback function to create a logical 3x3 grid.
3. Instruct the user to place a basketball sticker in the grid.
4. Use YOLO to detect the basketball sticker's bounding box and update the grid.
5. Check if the human player has won or if the grid is full; if not, continue the game.
6. Determine the robot player’s move using a simple algorithm.
7. Prompt the user to place an apple sticker on the pump and move the pump to the grid position.
8. Update the grid with the robot player’s move.
9. Check if the robot player has won or if the grid is full; if not, continue the game.
10. Repeat the steps until the game is over, and print the grid and result on the screen.
