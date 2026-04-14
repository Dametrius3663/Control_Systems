from picarx import Picarxcd
import time
import cv2
import yaml
import numpy as np

px = Picarx()

# -----------------------------
# Camera Setup (OpenCV)
# -----------------------------
cap = cv2.VideoCapture(0)

# Load calibration data from Vision folder
try:
    with open('Vision/calibration_params.yml', 'r') as f:
        calibration_data = yaml.safe_load(f)
    
    cam_matrix = np.array(calibration_data['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(calibration_data['dist_coeff'], dtype=np.float32)
    
    # Use focal length from camera matrix
    FOCAL_LENGTH = cam_matrix[0][0]  # fx from camera matrix
    print(f"Loaded calibration data. Focal length: {FOCAL_LENGTH}")
    
except FileNotFoundError:
    print("Warning: calibration_params.yml not found in Vision folder. Using default values.")
    FOCAL_LENGTH = 500  # fallback
    cam_matrix = None
    dist_coeffs = None

# CHANGE THIS AFTER CALIBRATION
KNOWN_WIDTH = 10.0  # cm


def camera_distance_test(frame):
    # Undistort the frame if calibration data is available
    if cam_matrix is not None and dist_coeffs is not None:
        frame = cv2.undistort(frame, cam_matrix, dist_coeffs)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)

            if w > 0:
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

                cv2.putText(frame, f"{distance:.2f} cm",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                return distance

    return None


try:
    while True:
        print("\n--- SENSOR READINGS ---")

        # -----------------------------
        # 1. Ultrasonic Sensor
        # -----------------------------
        distance = px.ultrasonic.read()
        print(f"Ultrasonic Distance: {distance:.2f} cm")

        # -----------------------------
        # 2. Grayscale Sensor
        # -----------------------------
        grayscale_values = px.get_grayscale_data()
        print(f"Grayscale Values: {grayscale_values}")

        # Simple interpretation
        if min(grayscale_values) < 500:
            print("Line Detected (dark surface)")
        else:
            print("Light Surface")

        # -----------------------------
        # 3. Camera (OpenCV)
        # -----------------------------
        ret, frame = cap.read()
        if ret:
            cam_distance = camera_distance_test(frame)

            if cam_distance:
                print(f"Camera Estimated Distance: {cam_distance:.2f} cm")
            else:
                print("Camera: No object detected")

            cv2.imshow("PiCar-X Camera Test", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopping test...")

finally:
    cap.release()
    cv2.destroyAllWindows()
