# -*- coding: utf-8 -*-
import matplotlib.image as mpimg
import numpy as np
import os
import io
import cv2
import math
import pickle
import sys
import time
from moviepy.editor import VideoFileClip
from skimage.transform import resize
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)

cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)
font = cv2.FONT_HERSHEY_SIMPLEX


# Global 함수 초기화
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)
l_center, r_center, lane_center = ((0,0)), ((0,0)), ((0,0))
pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
pts = pts.reshape((-1, 1, 2))

first_frame = 1
def grayscale(img):
    """Applies the Grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
        """Applies an image mask."""
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255, ) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        # vertiecs로 만든 polygon으로 이미지의 ROI를 정하고 ROI 이외의 영역은 모두 검정색으로 정한다.

        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

r_x2, l_x1 = 0, 0


l_lane, r_lane = [], []

# draw_lines 함수 내에서 l_lane, r_lane 값 설정 부분 추가
def draw_lines(img, lines):
    global cache
    global first_frame
    global next_frame
    global r_x2, l_x1
    global l_lane, r_lane  # 이 줄을 추가하세요

    y_global_min = img.shape[0]
    y_max = img.shape[0]

    l_slope, r_slope = [], []

    # 나머지 부분은 변경되지 않음

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = get_slope(x1, y1, x2, y2)
                if slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append(line)
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)

        y_global_min = min(y1, y2, y_global_min)

    if (len(l_lane) == 0 or len(r_lane) == 0):
        return 1, 0, 0, 0, 0  # 오류 방지

    l_slope_mean = np.mean(l_slope, axis=0)
    r_slope_mean = np.mean(r_slope, axis=0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    # 나머지 부분은 변경되지 않음

    return 0, l_x1, l_x2, r_x1, r_x2


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    print(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_pts(flag=0):
    vertices1 = np.array([
        [350, 600],
        [550, 370],
        [730, 370],
        [1100, 600]
                ])

    vertices2 = np.array([
                [0, 720],
                [210, 200],
                [970, 200],
                [1280, 720]
    ])
    if flag == 0 : return vertices1
    if flag == 1 : return vertices2
def process_image(image):
    global first_frame
    if image is None:
        print("Error: Input image is empty.")
        return None, None  # 또는 적절한 오류 처리를 추가
    resized_image = cv2.resize(image, (1280, 720))

    kernel_size = 3
    low_thresh = 100
    high_thresh = 150
    rho = 4
    theta = np.pi/180
    thresh = 100
    min_line_len = 50
    max_line_gap = 150

    gray_image = grayscale(resized_image)
    img_hsv = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 100, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)
    canny_edges = canny(gauss_gray, low_thresh, high_thresh)

    vertices = [get_pts(flag=0)]
    roi_image = region_of_interest(canny_edges, vertices)

    lines = cv2.HoughLinesP(roi_image, rho, theta, thresh, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    if lines is not None and len(lines) > 0:
        result, _, _, _, _ = draw_lines(resized_image, lines)
        result = visualize(result, l_x1, r_x2)
        return result, resized_image

    return resized_image, resized_image



def get_roi(image, canny_edges):
    vertices = [get_pts(image)]
    roi_image = region_of_interest(canny_edges, vertices)
    return roi_image

def visualize(result, l_x1, r_x2):
    height, width = result.shape[:2]
    length = 30
    thickness = 3
    whalf = int(width / 2)
    hhalf = int(height / 2)
    sl_color = yellow


    # Standard Line
    cv2.line(result, (whalf, lane_center[1]), (whalf, int(height)), sl_color, 2)
    cv2.line(result, (whalf, lane_center[1]), (lane_center[0], lane_center[1]), sl_color, 2)

    # Warning Boundary
    gap = 20 # 이 값도 나중에 오차 += 5m 정도 하려면 조절해야 할 듯
    length2 = 10
    wb_color = white
    cv2.line(result, (whalf - gap, lane_center[1] - length2), (whalf - gap, lane_center[1] + length2), wb_color, 1)
    cv2.line(result, (whalf + gap, lane_center[1] - length2), (whalf + gap, lane_center[1] + length2), wb_color, 1)

    # Lane Position
    lp_color = red
    cv2.line(result, (l_center[0], l_center[1]), (l_center[0], l_center[1] - length), lp_color, thickness)
    cv2.line(result, (r_center[0], r_center[1]), (r_center[0], r_center[1] - length), lp_color, thickness)
    cv2.line(result, (lane_center[0], lane_center[1]), (lane_center[0], lane_center[1] - length), lp_color, thickness)

    # 중앙 오프셋 및 차선 곡률에 대한 정보 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    lane_width_meters = 24.5  # 예시로 사용한 차선의 폭
    lane_width_pixels = r_x2 - l_x1  # 여기에 코드에서 계산한 차선의 픽셀 폭을 넣어야 합니다.
    xm_per_pix = lane_width_meters / lane_width_pixels

    # cv2.rectangle(result, (0,0), (400, 250), deepgray, -1)
    hei = 30
    font_size = 2

    # WARNING 및 방향 텍스트 추가
    if lane_center[0] == whalf:
        cv2.putText(result, 'Perfect : ', (450, hei+250), font, 1, white, font_size)
        cv2.putText(result, '100%', (620, hei+250), font, 1, white, font_size)
    else:
        if lane_center[0] < whalf - gap:
            cv2.putText(result, 'WARNING : ', (450, hei+250), font, 1, red, font_size)
            cv2.putText(result, 'Turn Left {:.2f}cm'.format(np.abs(lane_center[0] - whalf) * xm_per_pix), (620, hei+250), font, 1, red, font_size)
        elif lane_center[0] > whalf + gap:
            cv2.putText(result, 'WARNING : ', (450, hei+250), font, 1, red, font_size)
            cv2.putText(result, 'Turn Right {:.2f}cm'.format(np.abs(lane_center[0] - whalf) * xm_per_pix), (620, hei+250), font, 1, red, font_size)
        elif lane_center[0] < whalf:
            cv2.putText(result, 'STABLE : ', (450, hei+250), font, 1, white, font_size)
            cv2.putText(result, 'Turn Left {:.2f}cm'.format(np.abs(lane_center[0] - whalf) * xm_per_pix), (620, hei+250), font, 1, white, font_size)
        elif lane_center[0] > whalf:
            cv2.putText(result, 'STABLE : ', (450, hei+250), font, 1, white, font_size)
            cv2.putText(result, 'Turn Right {:.2f}cm'.format(np.abs(lane_center[0] - whalf) * xm_per_pix), (620, hei+250), font, 1, white, font_size)
        return result

def Region(image, l_x1, r_x2):
    # if l_x1 < 0 or r_x2 >= image.shape[1]:
    #      return image
    height, width = image.shape[:2]
    zeros = np.zeros_like(image)
    mask = cv2.fillPoly(zeros, [pts], lime)
    result = cv2.addWeighted(image, 1, mask, 0.3, 0)

    hhalf = int(height/2)


    if not lane_center[1] < hhalf:
        mask = visualize(result, l_x1, r_x2)
    return result

# webcam = cv2.VideoCapture(0).
#
# while webcam.isOpened():
#     _, frame = webcam.read()
#     if frame is None :
#         print("Error:Unable to read frame. Exiting...")
#         break
#     resized_frame = cv2.resize(frame, (1280, 720))
#     result, _ = process_image(resized_frame)
#     #edges = canny(frame, low_threshold=30, high_threshold=80)
#
#     result = cv2.resize(result, (1280, 720))
#     result = Region(result, l_x1, r_x2)
#     cv2.imshow("test", result)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# webcam.release()
# cv2.destroyAllWindows()
"""--------------------------Video test-------------------------"""
def save_video(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 60.0, (1280,720))
    return out

first_frame = 1

image_name = "test_videos/resized_bot1.mp4"
#cap = cv2.VideoCapture(image_name)
#clip1 = save_video('out_videos/result_bot1.mp4') # result 영상 저장
cap = cv2.VideoCapture(1)
while (cap.isOpened()):
    _, frame = cap.read()
    if frame is None :
        print("Error:Unable to read frame. Exiting...")
        break
    resized_frame = cv2.resize(frame, (1280, 720))
    result, _ = process_image(resized_frame)
    #edges = canny(frame, low_threshold=30, high_threshold=80)

    result = cv2.resize(result, (1280, 720))
    result = Region(result, l_x1, r_x2)
    cv2.imshow("result", result)
    #clip1.write(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#clip1.release()
cv2.destroyAllWindows()

"""--------------------------------------------------------------------------"""


# image_path = "drive.jpg"
#
# # 이미지 읽기
# image = cv2.imread(image_path)
#
# # Lane_Detection 함수를 사용하여 이미지 처리
# result = Lane_Detection(image)
# result = cv2.resize(result, (1280, 720))
# # 처리된 이미지를 윈도우에 표시
# cv2.imshow("Result", result)
#
# # 키보드 입력 대기
# cv2.waitKey(0)
#
# # 윈도우 닫기
# cv2.destroyAllWindows()
