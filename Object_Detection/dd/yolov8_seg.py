import numpy as np
import cv2
from yolo_segmentation import YOLOSegmentation
import time
# Segmentation detector
ys = YOLOSegmentation("best-seg.pt")

prevTime = 0
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX


cap = cv2.VideoCapture(1)

def calculate_angle_between_lines(line1, line2):
    # 두 선의 방향 벡터 계산
    vector1 = np.array([line1[2] - line1[0], line1[3] - line1[1]])
    vector2 = np.array([line2[2] - line2[0], line2[3] - line2[1]])

    # 두 벡터의 크기 계산
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # 두 벡터의 내적 계산
    dot_product = np.dot(vector1, vector2)

    # 각도 계산 (라디안에서 도로 변환)
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))

    # 라디안을 도로 변환
    angle_deg = np.degrees(angle_rad)

    return angle_deg
def webcam_FPS_output(frame,prevTime):
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
    cv2.putText(frame, str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    return prevTime

real_h = 11.2  # cm

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    whalf = width // 2
    hhalf = height // 2
    # object_detector 함수로 검출 결과 얻기
    bboxes, classes, segmentations, scores = ys.detect(frame)

    cv2.circle(frame, (whalf, hhalf), 2, (255, 0, 0), -1)

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        #print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        (x, y, x2, y2) = bbox
        w = x2-x
        h = y2- y
        #print(seg)
        cv2.circle(frame, (int((x+x2)/2), int((y+y2)/2)), 5, (0, 255, 0), -1) # 마커의 가운데 점
        #cv2.circle(frame, (int((x + x2) / 2), int(y)), 5, (0, 255, 0), -1)      # 마커의 위의 좌표
        #cv2.circle(frame, (int((x + x2) / 2), int(y2)), 5, (0, 255, 0), -1)     # 마커의 아래 좌표
        cv2.line(frame, (int((x + x2) / 2), int(y)),(int((x + x2) / 2), int(y2)),(0,0,255),2)
        # 확률 표시cv2.putText(frame, f"{score:.2f}", (235, 250), FONTS, 5, GREEN, 2)
        h = y2 - y
        # line_length = h  # 선의 길이 조절
        # line_thickness = 2  # 선의 두께 조절
        # line_color = (0, 255, 0)  # 선의 색상 조절
        #
        # # 선의 시작점과 끝점 계산
        # line_start = (whalf,hhalf - line_length // 2)
        # line_end = (whalf,hhalf + line_length // 2)
        #
        # # (whalf, hhalf)을 중심으로 짧은 선 그리기
        # cv2.line(frame, line_start, line_end, line_color, line_thickness)
        h_per_pixel = (real_h / h)
        distance = round((hhalf-(y+y2)/2) * h_per_pixel,2)
        cv2.rectangle(frame, (30, 30), (210, 60), BLACK, -1)
        if distance < 0:
            text = f'Move Above: {abs(distance)} cm'
            cv2.putText(frame, text, (35, 50), FONTS, 0.48, GREEN, 2)
        elif distance > 0:
            text = f'Move Below: {abs(distance)} cm'
            cv2.putText(frame, text, (35, 50), FONTS, 0.48, GREEN, 2)
        elif distance == 0:
            cv2.putText(frame, 'Perfect Center', (35, 50), FONTS, 0.48, GREEN, 2)
        #print(seg)
        # 바운딩 박스 사각형 cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
        #cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
        #cv2.rectangle(frame, (whalf-(w//2), hhalf-(h//2)), (whalf+(w//2), hhalf+(h//2)), (0, 255, 0), 2) # 화면 가운데에 표시

        # Extract x and y coordinates from the seg list
        x_values = seg[:, 0]
        y_values = seg[:, 1]
        sorted_indices_y = np.argsort(y_values)
        sorted_indices_x = np.argsort(x_values)

        second_max_index_y = seg[sorted_indices_y[-2]]
        second_min_index_x = seg[sorted_indices_x[1]]
        second_max_index_x = seg[sorted_indices_x[-2]]
        second_min_index_y = seg[sorted_indices_y[1]]

        # x 좌표에 대한 최소 및 최대 값 찾기
        if seg[sorted_indices_x[0]][1] < seg[sorted_indices_x[1]][1] :
            min_x_point = seg[sorted_indices_x[0]]
        else :
            min_x_point = seg[sorted_indices_x[1]]

        if seg[sorted_indices_x[-1]][1] < seg[sorted_indices_x[-2]][1] :
            max_x_point = seg[sorted_indices_x[-2]]
        else :
            max_x_point = seg[sorted_indices_x[-1]]

        if seg[sorted_indices_y[1]][0] < seg[sorted_indices_y[0]][0] :
            min_y_point = seg[sorted_indices_y[0]]
        else :
            min_y_point = seg[sorted_indices_y[1]]

        if seg[sorted_indices_y[-1]][0] < seg[sorted_indices_y[-2]][0] :
            max_y_point = seg[sorted_indices_y[-1]]
        else :
            max_y_point = seg[sorted_indices_y[-2]]

        # min__x_point = seg[sorted_indices_x[1]]
        # min__y_point = seg[sorted_indices_y[1]]
        # max__x_point = seg[sorted_indices_x[-2]]
        # max__y_point = seg[sorted_indices_y[-2]]
        if seg[sorted_indices_x[0]][1] < seg[sorted_indices_x[1]][1] :
            min__x_point = seg[sorted_indices_x[1]]
        else :
            min__x_point = seg[sorted_indices_x[0]]

        if seg[sorted_indices_x[-1]][1] < seg[sorted_indices_x[-2]][1] :
            max__x_point = seg[sorted_indices_x[-1]]
        else :
            max__x_point = seg[sorted_indices_x[-2]]

        if seg[sorted_indices_y[1]][0] < seg[sorted_indices_y[0]][0] :
            min__y_point = seg[sorted_indices_y[1]]
        else :
            min__y_point = seg[sorted_indices_y[0]]

        if seg[sorted_indices_y[-1]][0] < seg[sorted_indices_y[-2]][0] :
            max__y_point = seg[sorted_indices_y[-2]]
        else :
            max__y_point = seg[sorted_indices_y[-1]]

        # 최소 x 좌표에 점 찍기
        # cv2.circle(frame, (int(min_x_point[0]), int(min_x_point[1])), 5, (255, 0, 0), -1)
        #
        # # 최대 x 좌표에 점 찍기
        # cv2.circle(frame, (int(max_x_point[0]), int(max_x_point[1])), 5, (0, 255, 0), -1)
        #
        # # 최소 y 좌표에 점 찍기
        # cv2.circle(frame, (int(min_y_point[0]), int(min_y_point[1])), 5, (0, 0, 255), -1)
        #
        # # 최대 y 좌표에 점 찍기
        # cv2.circle(frame, (int(max_y_point[0]), int(max_y_point[1])), 5, (0, 255, 255), -1)

        # 중심점 찍기
        center_x1 = int((min_x_point[0] + min_y_point[0]) / 2)
        center_y1 = int((min_x_point[1] + min_y_point[1]) / 2)
        center_x2 = int((max_x_point[0] + max_y_point[0]) / 2)
        center_y2 = int((max_x_point[1] + max_y_point[1]) / 2)

        #cv2.line(frame,(center_x1, center_y1),(center_x2, center_y2), (0,155,0),2)

        center__x1 = int((max__x_point[0] + min__y_point[0]) / 2)
        center__y1 = int((max__x_point[1] + min__y_point[1]) / 2)
        center__x2 = int((min__x_point[0] + max__y_point[0]) / 2)
        center__y2 = int((min__x_point[1] + max__y_point[1]) / 2)
        #cv2.circle(frame, (center__x1, center__y1), 5, (255, 255, 255), -1)
        #cv2.circle(frame, (center__x2, center__y2), 5, (255, 255, 255), -1)

        cv2.line(frame, (int((x + x2) / 2), int(y)), (int((x + x2) / 2), int(y2)), (0, 0, 255), 2)
        # 바운딩 박스의 중앙선.
        line0 = [center_x1, center_y1, center_x2, center_y2]
        line1 = [center__x1, center__y1, center__x2, center__y2]
        line2 = [int((x + x2) / 2), int(y), int((x + x2) / 2), int(y2)]

        # 각도 계산
        angle_deg_line0_line2 = calculate_angle_between_lines(line0, line2)
        angle_deg_line1_line2 = calculate_angle_between_lines(line1, line2)
        if angle_deg_line0_line2 <= 1 :
            cv2.line(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 155, 0), 2)
            text = '100% degree'
            cv2.putText(frame, text, (75, 100), FONTS, 0.48, BLACK, 2)
        elif angle_deg_line0_line2 <= angle_deg_line1_line2 :
            cv2.line(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 155, 155), 2)
            text = f'Move Right: {int(angle_deg_line0_line2)} degree'
            cv2.putText(frame, text, (75, 100), FONTS, 0.48, BLACK, 2)
        elif angle_deg_line0_line2 >= angle_deg_line1_line2:
            cv2.line(frame, (center__x1, center__y1), (center__x2, center__y2), (0, 155, 0), 2)
            text = f'Move Left: {int(angle_deg_line1_line2)} degree'
            cv2.putText(frame, text, (75, 100), FONTS, 0.48, BLACK, 2)

        # 결과 출력
        print(angle_deg_line0_line2)
        print(angle_deg_line1_line2)

    # center_seg = np.array(seg)
        # # print(center_seg)
        # # print(type(center_seg))
        #
        #
        # x_shift = whalf - (x + x2) // 2
        # y_shift = hhalf - (y + y2) // 2
        #
        # center_seg[:, 0] += x_shift
        # center_seg[:, 1] += y_shift
        #
        # cv2.polylines(frame, [center_seg], True, (0, 0, 255), 4)

    prevTime = webcam_FPS_output(frame,prevTime)

    cv2.imshow('Webcam Object Detection', frame)

    # 종료 조건 (q 키 누를 때 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()





