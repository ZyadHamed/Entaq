import keyboard
import cv2
import time
import mediapipe as mp
import copy
import torch
from torchvision import transforms
from PIL import Image
import rembg
import numpy as np

#Loading the model and preparing transformations with the  same parameters used during training
model = torch.load("torchmodel.pth")
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of the model
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Define the hands library
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
#mpDraw = mp.solutions.drawing_utils

#User_Specified_Variables (Tweak as needed)
GrayScale_Cropped_Image = False
RemoveBackground_Cropped_Image = False

#Public Variables
imgUnaltered = 0
fullChar = ""
pTime = 0
cTime = 0

#Launch the cv2 window
cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
cap = cv2.VideoCapture(0)
def DrawOnImage(img):
    imgUnaltered = copy.deepcopy(img)
    height, width, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #Basic maximum and minimum algorithm, intitalize max to 0 and min to maxmium value
    x_min = width
    x_max = 0
    y_min = height
    y_max = 0
    #Loop on the hands landmarks (1 hand for now)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): #Loop on each landmark
                #The actual x and y of the landmark (as opposed to the ratio outputted by the library)
                cx, cy = int(lm.x*width), int(lm.y*height)
                #Update the max and min values whenever needed
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
            side_length = 0
            startPoint = ()
            endPoint = ()
            #Form a square around the hand and maximize the area of that square
            if(y_max - y_min > x_max - x_min):
                # The hand is oriented vertically (the difference between the furthest y_coordinates is larger than that of the x_coordinates)
                #the Set the side_length of the square to the difference between the 2 y_coordinates and add 50 to include bits of the hand outside the landmarks
                side_length = y_max - y_min + 50
                #The remaining length to be added to the x_coordinates from both right and left to make the distance between them the same as the distance between y_coordinates (a square)
                margin_addition = (side_length - (x_max - x_min)) / 2
                #Add the margin_addition from left and right, include the 50 added to side_length in the y_coordinate calculation
                startPoint = (int(x_min - margin_addition), y_min - 25)
                endPoint = (int(x_max + margin_addition), y_max + 25)
            else:
                # The hand is oriented horizontally (the difference between the furthest x_coordinates is larger than that of the y_coordinates)
                #Similar algorithm as above
                side_length = x_max - x_min + 50
                margin_addition = (side_length - (y_max - y_min)) / 2
                startPoint = (x_min - 25, int(y_min - margin_addition))
                endPoint = (x_max + 25, int(y_max + margin_addition))
            #Make a rectangle from the estimated points that would make the best possible square
            cv2.rectangle(img, startPoint, endPoint, (0, 0, 0), 1)
            #Crop the image on that square and use this image for the classification task
            imgUnaltered = imgUnaltered[startPoint[1]:endPoint[1], startPoint[0]: endPoint[0]]
    return img, imgUnaltered

def PredictImage(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictImage = img.to(device)
    optimizedModel = model.to(device)

    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
               'I,', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    with torch.no_grad():  # No need to track gradients for inference
        outputs = optimizedModel(predictImage)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    annotated_img, cropped_img = DrawOnImage(img)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    if keyboard.is_pressed("q"):
        cv2.destroyAllWindows()
        break
    elif keyboard.is_pressed("e"):
        if RemoveBackground_Cropped_Image == True:
            cropped_img = rembg.remove(cropped_img)
        cropped_img = Image.fromarray(cropped_img).convert('RGB')
        if GrayScale_Cropped_Image == True:
            cropped_img.convert('L')
        image = transform(cropped_img)  # Apply the transformations
        image = image.unsqueeze(0)  # Add a batch dimension to be predictable
        prediction = PredictImage(image)
        fullChar += prediction
    elif keyboard.is_pressed("backspace"):
        fullChar = fullChar[: len(fullChar) - 1]

    cv2.putText(annotated_img, fullChar, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", annotated_img)
    cv2.waitKey(1)