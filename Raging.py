from picarx import Picarx
from time import sleep
import cv2
import numpy as np


px = Picarx()

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Camera parameters
camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))
marker_length = 0.05  # 5cm markers


def clamp_number(num, a, b):
    return max(min(num, max(a, b)), min(a, b))


def main():
    speed = 50
    dir_angle = 0
    x_angle = 0
    y_angle = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, rejected = detector.detectMarkers(frame)

        # Look for marker 1
        marker1 = None
        if ids is not None:
            for i, id_val in enumerate(ids.flatten()):
                if id_val == 1:
                    marker1 = corners[i]
                    break

        if marker1 is not None:
            # Get marker center
            coords = marker1[0]
            center_x = int(np.mean(coords[:, 0]))
            center_y = int(np.mean(coords[:, 1]))

            # Pan/tilt camera to track marker
            x_angle += (center_x * 10 / 640) - 5
            x_angle = clamp_number(x_angle, -35, 35)
            px.set_cam_pan_angle(x_angle)

            y_angle -= (center_y * 10 / 480) - 5
            y_angle = clamp_number(y_angle, -35, 35)
            px.set_cam_tilt_angle(y_angle)

            # Drive toward marker
            # Steering lags slightly behind camera angle to avoid confusion
            if dir_angle > x_angle:
                dir_angle -= 1
            elif dir_angle < x_angle:
                dir_angle += 1
            px.set_dir_servo_angle(x_angle)
            px.forward(speed)

        else:
            # No marker found - stop
            px.stop()

        sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    finally:
        px.stop()
        cap.release()
        print("stop and exit")
