import cv2

webcam = cv2.VideoCapture(1)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    result, resized_frame = process_image(frame)
    result = cv2.resize(result, (1280, 720))
    result = Region(result, l_x1, r_x2)

    if status:
        cv2.imshow("test", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()