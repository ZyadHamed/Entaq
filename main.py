import math
from playsound import playsound
import sys
import cv2
import time
import mediapipe as mp
import keyboard
import numpy as np
import multiprocessing
import winsound

#Calculating the distance between 2 points
def Distance(landmark1, landmark2, image):
    x1 = landmark1.x * image.shape[1]
    x2 = landmark2.x * image.shape[1]
    y1 = landmark1.y * image.shape[0]
    y2 = landmark2.y * image.shape[0]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) / 30
#Launch the cv2 window
cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
cap = cv2.VideoCapture(0)

#Define the hands library
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
fullChar = ""
#keyboard.on_press_key("r", lambda _:print("You pressed r"))
while True:
#Show the hand landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    last_data = 0
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        last_data = results.multi_hand_landmarks
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

#If q is pressed, check each hand sign
    if keyboard.is_pressed("q"):
        #elif Distance(last_data[0].landmark[5], last_data[0].landmark[9], img) <= 2 and Distance(last_data[0].landmark[9], last_data[0].landmark[13], img) <= 2 and Distance(last_data[0].landmark[13], last_data[0].landmark[17], img) <= 2 and Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) >= 5.5:
        #    print('B')
        if Distance(last_data[0].landmark[8], last_data[0].landmark[12], img) <= 1.5 and Distance(last_data[0].landmark[12], last_data[0].landmark[16], img) <= 1.5 and Distance(last_data[0].landmark[16], last_data[0].landmark[20], img) <= 1.5 and Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) >= 4.5:
            print('C')
            fullChar += 'C'
        elif Distance(last_data[0].landmark[7], last_data[0].landmark[11], img) <= 2 and Distance(last_data[0].landmark[11], last_data[0].landmark[15], img) <= 2 and Distance(last_data[0].landmark[15], last_data[0].landmark[19], img) <= 2 and 3 >= Distance(
                last_data[0].landmark[4], last_data[0].landmark[6], img) > 2 and Distance(last_data[0].landmark[4], last_data[0].landmark[14], img) > 2.1:
            fullChar += 'A'

        elif Distance(last_data[0].landmark[4], last_data[0].landmark[11], img) <= 1 and Distance(last_data[0].landmark[8], last_data[0].landmark[12], img) >= 5:
            fullChar += 'P'

        elif Distance(last_data[0].landmark[4], last_data[0].landmark[10], img) <= 1.5 and Distance(last_data[0].landmark[3], last_data[0].landmark[6], img) <= 1.5 and Distance(last_data[0].landmark[20], last_data[0].landmark[4], img) >= 3:
            fullChar += 'i'

        elif Distance(last_data[0].landmark[8], last_data[0].landmark[12], img) <= 1 and Distance(last_data[0].landmark[12], last_data[0].landmark[16], img) <= 1 and Distance(last_data[0].landmark[16], last_data[0].landmark[20], img) <= 1 and Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) <= 1.5:
            fullChar += 'o'

        elif Distance(last_data[0].landmark[7], last_data[0].landmark[11], img) <= 2 and Distance(last_data[0].landmark[11], last_data[0].landmark[15], img) <= 2 and Distance(last_data[0].landmark[15], last_data[0].landmark[19], img) <= 2 and 3 >= Distance(
                last_data[0].landmark[4], last_data[0].landmark[6], img) < 1.2:

            fullChar += 'S'

        elif Distance(last_data[0].landmark[18], last_data[0].landmark[14], img) >= 2.3 and Distance(last_data[0].landmark[14], last_data[0].landmark[10], img) <= 3 and Distance(last_data[0].landmark[10], last_data[0].landmark[6], img) <= 3 and Distance(last_data[0].landmark[4], last_data[0].landmark[6], img) >= 3:

            fullChar += 'm'

        elif Distance(last_data[0].landmark[6], last_data[0].landmark[10], img) > 1 and Distance(last_data[0].landmark[4], last_data[0].landmark[10], img) <= 2 and Distance(last_data[0].landmark[14], last_data[0].landmark[18], img) <= 2 and Distance(last_data[0].landmark[4], last_data[0].landmark[11], img) <= 3 and Distance(last_data[0].landmark[20], last_data[0].landmark[16], img) <= 1.5:

            fullChar += 'n'

        elif Distance(last_data[0].landmark[20], last_data[0].landmark[4], img) <= 1 and Distance(last_data[0].landmark[12], last_data[0].landmark[3], img) <= 2 and Distance(last_data[0].landmark[16], last_data[0].landmark[3], img) <= 1.5 and Distance(last_data[0].landmark[8], last_data[0].landmark[2], img) <= 1.5:

            fullChar += 'e'

        elif Distance(last_data[0].landmark[14], last_data[0].landmark[18], img) <= 1.5 and Distance(last_data[0].landmark[4], last_data[0].landmark[14], img) <= 1.2 and Distance(last_data[0].landmark[7], last_data[0].landmark[11], img) <= 1 and Distance(last_data[0].landmark[8], last_data[0].landmark[12], img) <= 1.5:

            fullChar += 'r'

#Reset the character
    elif keyboard.is_pressed("s"):
        fullChar = ""

#Exit
    elif keyboard.is_pressed("e"):
        sys.exit()

    #print(Distance(last_data[0].landmark[6], last_data[0].landmark[10], img) > 1, " ", Distance(last_data[0].landmark[4], last_data[0].landmark[10], img) <= 2 , " ", Distance(last_data[0].landmark[14], last_data[0].landmark[18], img) <= 1.5, " ", Distance(last_data[0].landmark[4], last_data[0].landmark[11], img) <= 3, " ", Distance(last_data[0].landmark[20], last_data[0].landmark[16], img) <= 1)
    cv2.putText(img, fullChar, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)