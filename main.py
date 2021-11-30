import math
from playsound import playsound
import cv2
import time
import mediapipe as mp
import keyboard
import numpy as np
import multiprocessing
import winsound

def Distance(landmark1, landmark2, image):
    x1 = landmark1.x * image.shape[1]
    x2 = landmark2.x * image.shape[1]
    y1 = landmark1.y * image.shape[0]
    y2 = landmark2.y * image.shape[0]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) / 30
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
fullChar = ""
#keyboard.on_press_key("r", lambda _:print("You pressed r"))
while True:
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

    if keyboard.is_pressed("q"):
        #elif Distance(last_data[0].landmark[5], last_data[0].landmark[9], img) <= 2 and Distance(last_data[0].landmark[9], last_data[0].landmark[13], img) <= 2 and Distance(last_data[0].landmark[13], last_data[0].landmark[17], img) <= 2 and Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) >= 5.5:
        #    print('B')
        if Distance(last_data[0].landmark[8], last_data[0].landmark[12], img) <= 1.5 and Distance(last_data[0].landmark[12], last_data[0].landmark[16], img) <= 1.5 and Distance(last_data[0].landmark[16], last_data[0].landmark[20], img) <= 1.5 and Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) >= 4.5:
            print('C')
            fullChar += 'C'
        #elif Distance(last_data[0].landmark[7], last_data[0].landmark[11], img) <= 2 and Distance(last_data[0].landmark[11], last_data[0].landmark[15], img) <= 2 and Distance(last_data[0].landmark[15], last_data[0].landmark[19], img) <= 2 and 3 >= Distance(
        #        last_data[0].landmark[4], last_data[0].landmark[6], img) > 1.2:
        #    print('A')
        #    fullChar += 'A'
        elif Distance(last_data[0].landmark[8], last_data[0].landmark[12], img) <= 1 and Distance(last_data[0].landmark[12], last_data[0].landmark[16], img) <= 1 and Distance(last_data[0].landmark[16], last_data[0].landmark[20], img) <= 1 and Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) <= 1.5:
            print('O')
            fullChar += 'O'
        elif Distance(last_data[0].landmark[7], last_data[0].landmark[11], img) <= 2 and Distance(last_data[0].landmark[11], last_data[0].landmark[15], img) <= 2 and Distance(last_data[0].landmark[15], last_data[0].landmark[19], img) <= 2 and 3 >= Distance(
                last_data[0].landmark[4], last_data[0].landmark[6], img) < 1.2:
            print('S')
            fullChar += 'S'
        elif Distance(last_data[0].landmark[18], last_data[0].landmark[14], img) >= 2.3 and Distance(last_data[0].landmark[14], last_data[0].landmark[10], img) <= 3 and Distance(last_data[0].landmark[10], last_data[0].landmark[6], img) <= 3 and Distance(last_data[0].landmark[4], last_data[0].landmark[6], img) >= 3:
            print('M')
            fullChar += 'M'
        elif Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) <= 1 and Distance(last_data[0].landmark[8], last_data[0].landmark[12], img) >= 3 and Distance(last_data[0].landmark[12], last_data[0].landmark[16], img) <= 1.6 and Distance(last_data[0].landmark[16], last_data[0].landmark[20], img) >= 2:
            fullChar = ""
            fullChar = ""
            fullChar = "Perfecto"


        print(Distance(last_data[0].landmark[4], last_data[0].landmark[8], img))
        print(Distance(last_data[0].landmark[14], last_data[0].landmark[10], img))
    elif keyboard.is_pressed("s"):
        fullChar = ""
    elif keyboard.is_pressed("w"):
        if Distance(last_data[0].landmark[7], last_data[0].landmark[11], img) <= 2 and Distance(last_data[0].landmark[11], last_data[0].landmark[15], img) <= 2 and Distance(last_data[0].landmark[15], last_data[0].landmark[19], img) <= 2 and 3 >= Distance(
                last_data[0].landmark[4], last_data[0].landmark[6], img) < 1.2 and Distance(last_data[1].landmark[7], last_data[1].landmark[11], img) <= 2 and Distance(last_data[1].landmark[11], last_data[1].landmark[15], img) <= 2 and Distance(last_data[1].landmark[15], last_data[1].landmark[19], img) <= 2 and 3 >= Distance(
                last_data[1].landmark[4], last_data[1].landmark[6], img) < 1.2:
            print('S')
            winsound.PlaySound('SUIII.wav', winsound.SND_FILENAME)
        elif Distance(last_data[0].landmark[4], last_data[1].landmark[4], img) < 1.2 and Distance(last_data[0].landmark[8], last_data[1].landmark[8], img) < 1.2 and Distance(last_data[0].landmark[4], last_data[0].landmark[8], img) > 2.3 and Distance(last_data[1].landmark[4], last_data[1].landmark[8], img) > 2.3:
            fullChar = 'Heart'
        elif Distance(last_data[0].landmark[8], last_data[1].landmark[8], img) < 1.2 and Distance(last_data[0].landmark[4], last_data[1].landmark[4], img) > 4:
            fullChar = 'Please'
    cv2.putText(img, fullChar, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)