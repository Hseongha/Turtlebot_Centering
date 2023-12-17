import cv2 as cv
import numpy as np
import math
# Distance constants
KNOWN_DISTANCE = (50)  #
PERSON_WIDTH = (40)  #
MOBILE_WIDTH = (3.0)  #
MARKER_WIDTH = 5.5
# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
fps = 0
prev_time = cv.getTickCount()

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(640, 480), scale=1 / 255, swapRB=True)


# object detecqtor funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    center = []
    distances = []
    height, width = image.shape[:2]

    whalf = width // 2
    hhalf = height // 2
    real_h = 11.2  # cm

    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]
        x,y,w,h, = box
        h_per_pixel = (real_h / h)
        # label = "%s : %f" % (class_names[classid], score), 확률

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, class_names[classid], (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        if classid == 11:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])


        elif classid == 67:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])

        # Find the center point of the box
        center_x = box[0] + box[2] // 2
        center_y = box[1] + box[3] // 2
        # Add the center point to the data_list
        center.append((center_x, center_y))
        cv.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
        # math.sqrt(abs(center_x - whalf)**2 + abs(center_y - hhalf)**2) 대각선 길이,
        # 좌우 센터링 돼있단 가정하에.
        distancess = round((hhalf - center_y) * h_per_pixel,2)

        #print(distancess)
        distances.append(distancess)
    cv.circle(image, (whalf,hhalf), 5, (255, 0, 0), -1)
    print(data_list)
    return data_list, distances


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


# reading the reference image from dir
ref_person = cv.imread('ReferenceImages/person.jpg')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data,_ = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]
 # 이미지4, 14 기준으로 오브젝드 디텍션 한다음에 거리를 잰 값
person_data,_ = object_detector(ref_person)
person_width_in_rf = person_data[0][1]
# 픽셀 거리 재는 방법이구나..
print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

DISTANCE_THRESHOLD = 100  # 센티미터 단위

cap_source = 0 # 초기 비디오 소스
cap = cv.VideoCapture(cap_source)

while True:
    ret, frame = cap.read()
    # Resize the frame to match the calculation size
    frame = cv.resize(frame, (640, 480))

    data, dist = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            x, y = d[2]
            # 거리가 임계값 미만인 경우
            for dis in dist:
                if dis > 1000:
                    # 현재 비디오 캡처를 해제
                    cap.release()

                    # 다음 비디오 소스로 전환
                    cap_source = 1 if cap_source == 0 else 0

                    # 새로운 비디오 캡처를 엽니다.
                    cap = cv.VideoCapture(cap_source)

                    # 루프를 빠져나가 새로운 비디오 소스를 처리하기 시작합니다.
                    break


            # Check if the person is above or below the center



        elif d[0] == 'cell phone':
            # distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'stop sign':
            x, y = d[2]
            for dis in dist :
                continue

        cv.rectangle(frame, (30, 30), (210, 60), BLACK, -1)
        if dis < 0 :
            text = f'Move Above: {abs(dis)} cm'
            cv.putText(frame, text, (35, 50), FONTS, 0.48, GREEN, 2)
        elif dis > 0:
            text = f'Move Below: {abs(dis)} cm'
            cv.putText(frame, text, (35,  50), FONTS, 0.48, GREEN, 2)
        elif dis == 0 :
            cv.putText(frame, 'Perfect Center', (35,  50), FONTS, 0.48, GREEN, 2)
        # FPS 계산 및 출력
    current_time = cv.getTickCount()
    time_diff = (current_time - prev_time) / cv.getTickFrequency()
    fps = 1 / time_diff
    prev_time = current_time

    # FPS 정보를 화면에 출력
    cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), FONTS, 1, GREEN, 2)

    # 화면에 이미지 표시
    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()


