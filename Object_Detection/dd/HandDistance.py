import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
FONTS = cv2.FONT_HERSHEY_COMPLEX
model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        #label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, class_names[classid], (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 67:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
            print(box)
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list

# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
print(coff)
# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Game Variables
cx, cy = 250, 250
color = (255, 0, 255)

# Loop
while True:
    success, img = cap.read()
    hands, img= detector.findHands(img, draw=False)

    if hands:
        lmList = hands[0]['lmList']
        center = (hands[0]['center'])
        x, y, w, h = hands[0]['bbox']
        x1, y1, z1 = lmList[5]
        x2, y2, z2 = lmList[17]
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff    # coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
        distanceCM = A * distance ** 2 + B * distance + C

        #print(distanceCM, distance)


        if distanceCM < 40 :
            #if x < cx < x+w and y < cy < y+h :
            color = (0,255,0)
        else :
            color = (255,0,255)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+5, y-10))
    # Draw Button
    cv2.circle(img, (cx, cy), 30, color, cv2.FILLED)
    cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
    cv2.circle(img, (cx, cy), 20, (255, 255, 255), 2)
    cv2.circle(img, (cx, cy), 30, (50, 50, 50), 2)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
