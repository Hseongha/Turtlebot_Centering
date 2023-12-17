import cv2 as cv
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation

import time

# 웹캠 설정
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# YOLO 모델 초기화
model = YOLO('best-seg.pt')

# FPS 측정을 위한 변수 초기화
prev_time = 0
fps = 0
# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX
class_names = []
prevTime = 0
def object_detector(image):
    # 모델이 이미 초기화된 것으로 가정
    results = model(image)
    height, width = image.shape[:2]
    xyxy = []
    xywh = []
    whalf = width // 2
    hhalf = height // 2
    real_h = 11.2  # cm

    for result in results[0]:
        if len(result.boxes) > 0 :
            boxes = result.boxes.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            confidence_value = confidences[0]
            cv.putText(frame, f"{confidence_value:.2f}", (235, 250), FONTS, 5, GREEN, 2)


            xyxy = result.boxes.xyxy.cpu().numpy()
            xywh = result.boxes.xywh.cpu().numpy()
            #print(xywh[0][1])
            cv.circle(image, (int(xywh[0][0]),int(xywh[0][1])), 5, (0, 255, 0), -1)
            h = xywh[0][3]
            h_per_pixel = (real_h/h)
            distance = round((hhalf-xywh[0][1]) * h_per_pixel,2)
            cv.rectangle(frame, (30, 30), (210, 60), BLACK, -1)
            if distance < 0:
                text = f'Move Above: {abs(distance)} cm'
                cv.putText(frame, text, (35, 50), FONTS, 0.48, GREEN, 2)
            elif distance > 0:
                text = f'Move Below: {abs(distance)} cm'
                cv.putText(frame, text, (35, 50), FONTS, 0.48, GREEN, 2)
            elif distance == 0:
                cv.putText(frame, 'Perfect Center', (35, 50), FONTS, 0.48, GREEN, 2)
        else:
            xyxy=[[0,0,0,0]]

        cv.circle(image, (whalf, hhalf), 5, (255, 0, 0), -1)
    return xyxy,xywh

def webcam_FPS_output(frame, prevTime):
    curTime = time.time()

    sec = curTime - prevTime
    prevTime = curTime

    fps = 1 / (sec)

    print
    "Time {0} ".format(sec)
    print
    "Estimated fps {0} ".format(fps)

    # 프레임 수를 문자열에 저장
    str = "FPS: %0.1f" % fps
    cv.putText(frame, str, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    return prevTime

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    # object_detector 함수로 검출 결과 얻기
    xyxy,xywh = object_detector(frame)

    # 검출 결과를 화면에 표시
    for item in xyxy:
        x1, y1, x2, y2 = xyxy[0]
        # 각 검출된 객체에 대한 처리 (예: 프레임에 사각형 그리기)
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #cv.polylines(frame, [seg], True, (0, 0, 255), 4)
        # cv.putText(frame, f'{class_name} {confidence:.2f}', (int(x), int(y) - 10),
        #             cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 이미지 표시
    prevTime = webcam_FPS_output(frame, prevTime)
    cv.imshow('Webcam Object Detection', frame)

    # 종료 조건 (q 키 누를 때 종료)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# 웹캠 해제 및 창 닫기
cap.release()
cv.destroyAllWindows()
