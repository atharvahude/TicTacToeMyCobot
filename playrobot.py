import cv2
import json
from ultralytics import YOLO
from pymycobot.mycobot import MyCobot
import time

MODEL_PATH = 'yolov8_custom.pt'
ANGLES_PATH = 'coordinates.json'
CV2_CAMERA_NUMBER = 2
COM_PORT = "COM11"

mc = MyCobot(COM_PORT, 115200)
with open(ANGLES_PATH) as f:
   REAL_WORLD_ANGLES = json.load(f)

import random
random.seed(69)

def drawBoard(board):
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('   |   |')

def playAgain():
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')

def makeMove(board, letter, move):
    board[move] = letter

def isWinner(bo, le):
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or
    (bo[4] == le and bo[5] == le and bo[6] == le) or 
    (bo[1] == le and bo[2] == le and bo[3] == le) or 
    (bo[7] == le and bo[4] == le and bo[1] == le) or 
    (bo[8] == le and bo[5] == le and bo[2] == le) or 
    (bo[9] == le and bo[6] == le and bo[3] == le) or 
    (bo[7] == le and bo[5] == le and bo[3] == le) or 
    (bo[9] == le and bo[5] == le and bo[1] == le)) 

def getBoardCopy(board):
    dupeBoard = []
    for i in board:
        dupeBoard.append(i)
    return dupeBoard

def isSpaceFree(board, move):
    return board[move] == ' '

def getPlayerMove(board):
    move = ' '
    while move not in '1 2 3 4 5 6 7 8 9'.split() or not isSpaceFree(board, int(move)):
        print('What is your next move? (1-9)')
        move = input()
    return int(move)

def chooseRandomMoveFromList(board, movesList):
    possibleMoves = []
    for i in movesList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)
    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None

def getComputerMove(board, computerLetter):
    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, computerLetter, i)
            if isWinner(copy, computerLetter):
                return i
            
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i

    move = chooseRandomMoveFromList(board, [1, 3, 7, 9])
    if move != None:
        return move

    if isSpaceFree(board, 5):
        return 5
    
    return chooseRandomMoveFromList(board, [2, 4, 6, 8])

def isBoardFull(board):
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True




class CaptureROI():
    def __init__(self) -> None:
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.roi_coordinates = []
        self.img = None

    def draw_rectangle(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.roi_coordinates.append((self.ix, self.iy, x, y))
            cv2.imshow('image', self.img)

    def get_roi(self):
        cam = cv2.VideoCapture(CV2_CAMERA_NUMBER)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        result, image = cam.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        if result:
            print('Success Catured Image')
        self.img= image

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_rectangle)
        while True:
            cv2.imshow('image', self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        return self.roi_coordinates
    
    def crop_resize_and_save(self,image_path):
        (x1, y1,x2, y2) = self.roi_coordinates[0]
        self.cropped_image = self.img[y1:y2, x1:x2]
        self.cropped_image = cv2.resize(self.cropped_image, (501, 501), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(image_path, self.cropped_image)
        return self.cropped_image
    
    def get_cropped_camera_input(self):
        cam = cv2.VideoCapture(CV2_CAMERA_NUMBER) 
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        result, image = cam.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        if result:
            print('Success Catured Image')
        self.img= image

        (x1, y1,x2, y2) = self.roi_coordinates[0]
        self.cropped_image = self.img[y1:y2, x1:x2]
        self.cropped_image = cv2.resize(self.cropped_image, (501, 501), interpolation = cv2.INTER_LINEAR)
        return self.cropped_image

def get_matrix_block(detections):
    all_pos = []
    H33,W33 = (167.0, 167.0)
    for det in detections:
        center_x = (det[0] + det[2]) / 2
        center_y = (det[1] + det[3]) / 2
        x,y = 0,0
        for i in range(1,4):
            if center_x == min(center_x,W33*i) :
                x = i
                break
        for i in range(1,4):
            if center_y == min(center_y,H33*i) :
                y = i
                break
        if y == 3:
            all_pos.append(x)
        if y == 2:
            all_pos.append(3+x)
        if y == 1:
            all_pos.append(6+x)      
    return all_pos

def getPlayerInputNumber(theBoard,user_input_indices):
    '''
    Checks the list and finds the new unique entry
    '''
    for i in user_input_indices:
        if theBoard[i] == ' ':
            return i

def pump_on():
    mc.set_basic_output(2, 0)
    mc.set_basic_output(5, 0)

def pump_off():
    mc.set_basic_output(2, 1)    
    mc.set_basic_output(5, 1)

def place_marker(position):
    mc.send_angles(REAL_WORLD_ANGLES["0"], 50)
    time.sleep(2)
    _ = input('Place Marker')
    pump_on()
    time.sleep(2)
    mc.send_angles(REAL_WORLD_ANGLES[str(position)], 50)
    time.sleep(2)
    pump_off()
    time.sleep(2)
    mc.send_angles(REAL_WORLD_ANGLES["0"], 50)
    time.sleep(2)

class Inferyolo():
    def __init__(self) -> None:
        self.model = YOLO(MODEL_PATH)
    def downstream(self,crop_image):
        results = self.model(crop_image)
        detections = results[0].boxes.xyxy.numpy().tolist()
        user_input_indexes = get_matrix_block(detections)
        return user_input_indexes

if __name__ == '__main__':
    print('Welcome to Tic Tac Toe!')
    roi_capture = CaptureROI()
    infer = Inferyolo()
    roi_coordinates = roi_capture.get_roi()
    print("ROI Coordinates: ",roi_coordinates)
    crop_image = roi_capture.crop_resize_and_save('crop.png')

    while True:
        theBoard = [' '] * 10
        playerLetter, computerLetter = ['X', 'O']
        turn = 'player'
        print('The ' + turn + ' will go first.')
        gameIsPlaying = True
        
        while gameIsPlaying:
            if turn == 'player':
                drawBoard(theBoard)
                input_image_cropped = roi_capture.get_cropped_camera_input('')
                user_input_indices = infer.downstream(input_image_cropped)
                print(user_input_indices,'      ',theBoard)
                move = getPlayerInputNumber(theBoard,user_input_indices)
                _ = input('Press enter')
                if move == None:
                    continue
                makeMove(theBoard, playerLetter, move)
                if isWinner(theBoard, playerLetter):
                    drawBoard(theBoard)
                    print('Hooray! You have won the game!')
                    gameIsPlaying = False
                else:
                    if isBoardFull(theBoard):
                        drawBoard(theBoard)
                        print('The game is a tie!')
                        break
                    else:
                        turn = 'computer'
            else:
                move = getComputerMove(theBoard, computerLetter)
                makeMove(theBoard, computerLetter, move)
                place_marker(move)
                if isWinner(theBoard, computerLetter):
                    drawBoard(theBoard)
                    print('The computer has beaten you! You lose.')
                    gameIsPlaying = False
                else:
                    if isBoardFull(theBoard):
                        drawBoard(theBoard)
                        print('The game is a tie!')
                        break
                    else:
                        turn = 'player'
        if not playAgain():
            break