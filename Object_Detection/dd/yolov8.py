import cv2
from ultralytics import YOLO
import time

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# YOLO 모델 초기화
model = YOLO('best-seg1.pt')

# FPS 측정을 위한 변수 초기화
prev_time = 0
fps = 0

ret = True
while ret:
    ret, frame = cap.read()

    # YOLO 모델을 사용하여 객체 추적
    results = model.track(frame, persist=False)

    # 추적 결과를 이미지로 플로팅하여 가져옴
    frame_ = results[0].plot()

    # 현재 시간 기록
    curr_time = time.time()

    # FPS 계산
    if prev_time != 0:
        fps = 1 / (curr_time - prev_time)

    # FPS를 이미지에 표시
    cv2.putText(frame_, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 창에 이미지 표시
    cv2.imshow('frame', frame_)

    # 현재 시간을 이전 시간으로 업데이트
    prev_time = curr_time

    # 종료 조건
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
