import cv2
import numpy as np
from picarx import Picarx

px = Picarx()

cap = cv2.VideoCapture(0)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Control parameters
STEER_GAIN = 0.05
MAX_STEER = 30

TARGET_FACE_SIZE = 200  # desired width of face (pixels)
SPEED_FORWARD = 30
SPEED_BACKWARD = -20

def stop():
    px.forward(0)
    px.set_dir_servo_angle(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        h, w = frame.shape[:2]

        if len(faces) > 0:
            # Pick largest face (closest)
            face = max(faces, key=lambda f: f[2]*f[3])
            (x, y, fw, fh) = face

            face_center_x = x + fw / 2
            frame_center_x = w / 2

            error = face_center_x - frame_center_x

            # Steering
            steer = -error * STEER_GAIN
            steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))
            px.set_dir_servo_angle(steer)

            # Distance control
            if fw < TARGET_FACE_SIZE - 20:
                px.forward(SPEED_FORWARD)   # too far → move forward
            elif fw > TARGET_FACE_SIZE + 20:
                px.forward(SPEED_BACKWARD)  # too close → back up
            else:
                px.forward(0)               # perfect distance

            # Draw box
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0,255,0), 2)

            print(f"Face width: {fw}, Steer: {steer:.1f}")

        else:
            # No face → stop or slowly search
            stop()

        cv2.imshow("Face Follow", frame)
        if cv2.waitKey(1) == 27:
            break

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()