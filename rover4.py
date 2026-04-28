import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
from picarx import Picarx
from Vision.app.core.config import Config
import time

# -----------------------
# Hardware
# -----------------------
px = Picarx()
config = Config.get_instance()

aruco_dict = config.aruco_dict
aruco_params = config.aruco_params

camera_matrix = config.cam_matrix
dist_coeffs = config.dist_coeffs
marker_size = config.marker_size

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = aruco.ArucoDetector(aruco_dict, aruco_params)
marker_length = marker_size / 100

# -----------------------
# Control
# -----------------------
speed = 100
current_speed = 0
max_speed = speed
accel_step = 2

def update_speed(target_speed):
    global current_speed

    if current_speed < target_speed:
        current_speed += accel_step
    elif current_speed > target_speed:
        current_speed -= accel_step

    current_speed = max(0, min(current_speed, max_speed))
    return current_speed

# -----------------------
# STATE
# -----------------------
active_target = None
reverse_mode = False

# -----------------------
# FRAME LOCK SYSTEM (NEW)
# -----------------------
close_counter = 0
close_threshold_frames = 5   # must be close for 5 consecutive frames

# -----------------------
# SAFE STOP
# -----------------------
def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)

# -----------------------
# ACTIONS
# -----------------------
def AtMarker8():
    print("Marker 8 → VEER")
    px.set_dir_servo_angle(35)
    px.forward(update_speed(speed))

def AtMarker10():
    print("Marker 10 → TURN")
    px.set_dir_servo_angle(-45)
    px.forward(update_speed(speed))

# -----------------------
# TRACKING
# -----------------------
def track_marker_pnp(rvec, tvec, reverse=False):

    global active_target, close_counter

    tvec = np.array(tvec).reshape(3,)
    x = float(tvec[0])
    z = float(tvec[2])

    if abs(z) < 1e-6:
        close_counter = 0
        return

    steer = np.degrees(np.arctan2(x, z)) * 0.99

    if reverse:
        steer *= -1

    steer = float(np.clip(steer, -30, 30))

    px.set_dir_servo_angle(steer)
    px.forward(update_speed(speed))

    print(f"[TRACK] id:{active_target} x:{x:.2f} z:{z:.2f} steer:{steer:.2f} close_count:{close_counter}")

    # -----------------------
    # FRAME LOCK LOGIC (KEY FIX)
    # -----------------------

    if 0.2 < z < 0.5:
        close_counter += 1
    else:
        close_counter = 0

    if close_counter >= close_threshold_frames:
        print("LOCKED CLOSE → executing action")

        stop_car()

        target = active_target  # LOCK ID

        if target == 8:
            AtMarker8()
            time.sleep(1.0)
            stop_car()

        elif target == 10:
            AtMarker10()
            time.sleep(2.0)
            stop_car()

        close_counter = 0
        return "close"

# -----------------------
# MAIN LOOP
# -----------------------
def main(headless=False):

    global active_target, close_counter

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detector.detectMarkers(frame)

            # -----------------------
            # NO MARKER CASE
            # -----------------------
            if ids is None or len(ids) == 0:
                stop_car()
                active_target = None
                close_counter = 0
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )

            marker_map = {int(id_val): i for i, id_val in enumerate(ids.flatten())}

            # -----------------------
            # TARGET LATCH
            # -----------------------
            if active_target is None:
                if 10 in marker_map:
                    active_target = 10
                elif 8 in marker_map:
                    active_target = 8
                elif 11 in marker_map:
                    active_target = 11

            if active_target not in marker_map:
                active_target = None
                close_counter = 0
                continue

            i = marker_map[active_target]

            result = track_marker_pnp(
                rvecs[i],
                tvecs[i],
                reverse_mode
            )

            if result == "close":
                active_target = None
                close_counter = 0
                stop_car()

            # -----------------------
            # DISPLAY
            # -----------------------
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if not headless:
                cv2.imshow("Rover", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    finally:
        stop_car()
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete")


# -----------------------
# ENTRY
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main(args.headless)