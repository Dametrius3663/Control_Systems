import cv2
import numpy as np
from picarx import Picarx

px = Picarx()

cap = cv2.VideoCapture(0)

# Load face detector'
face_cascade = cv2.CascadeClassifier(
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
)
# Gains (tune these)
PAN_GAIN = 0.08
TILT_GAIN = 0.08

PAN_MAX = 45
TILT_MAX = 30

pan_angle = 0
tilt_angle = 0

def clamp(val, min_v, max_v):
    return max(min_v, min(max_v, val))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # pick largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])

        fx = x + fw // 2
        fy = y + fh // 2

        # errors
        err_x = fx - cx
        err_y = fy - cy

        # update servos
        pan_angle -= err_x * PAN_GAIN
        tilt_angle += err_y * TILT_GAIN

        pan_angle = clamp(pan_angle, -PAN_MAX, PAN_MAX)
        tilt_angle = clamp(tilt_angle, -TILT_MAX, TILT_MAX)

        px.set_cam_pan_angle(pan_angle)
        px.set_cam_tilt_angle(tilt_angle)

        # draw
        cv2.rectangle(frame, (x,y), (x+fw,y+fh), (0,255,0), 2)

    cv2.imshow("Face Track", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()