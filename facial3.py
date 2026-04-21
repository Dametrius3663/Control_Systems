import cv2
import numpy as np
from picarx import Picarx
import time

px = Picarx()

cap = cv2.VideoCapture(0)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
)

# -------------------
# Gains (tune these)
# -------------------
PAN_GAIN = 0.09
TILT_GAIN = 0.05
STEER_GAIN = 0.05
SPEED_GAIN = 1

# Limits
PAN_MAX = 45
TILT_MAX = 30
STEER_MAX = 45

# Target face size (distance control)
TARGET_FACE_WIDTH = 180

# State
pan_angle = 0
tilt_angle = 0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Pick largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        fx = x + fw // 2
        fy = y + fh // 2

        # -----------------------------
        # 1. HEAD TRACKING (pan/tilt)
        # -----------------------------
        err_x = fx - cx
        err_y = fy - cy

        if abs(err_x) < 5:
            err_x = 0
        if abs(err_y) < 15:
            err_y = 0

        alpha = 0.7

        pan_angle = alpha * pan_angle + (1 - alpha) * (pan_angle + err_x * PAN_GAIN)
        tilt_angle = alpha * tilt_angle + (1 - alpha) * (tilt_angle - err_y * TILT_GAIN)

        pan_angle = clamp(pan_angle, -PAN_MAX, PAN_MAX)
        tilt_angle = clamp(tilt_angle, -TILT_MAX, TILT_MAX)

        px.set_cam_pan_angle(pan_angle)
        px.set_cam_tilt_angle(tilt_angle)

        # -----------------------------
        # 2. STEERING
        # -----------------------------
        steer = clamp(err_x * STEER_GAIN, -STEER_MAX, STEER_MAX)
        px.set_dir_servo_angle(pan_angle)

        # -----------------------------
        # 3. SPEED
        # -----------------------------
        face_error = TARGET_FACE_WIDTH - fw

        if abs(face_error) < 20:
            speed = 0
        else:
            speed = face_error * SPEED_GAIN

        speed = clamp(speed, -30, 40)
        px.forward(speed)

        # Debug only (no display)
        print(
            f"FaceW:{fw} "
            f"ErrX:{err_x:.0f} Pan:{pan_angle:.1f} Tilt:{tilt_angle:.1f} "
            f"Steer:{steer:.1f} Speed:{speed:.1f}"
        )

    else:
        # No face → stop
        px.forward(0)
        print("No face detected")

    # Small delay to reduce CPU usage
    time.sleep(0.03)

cap.release()
