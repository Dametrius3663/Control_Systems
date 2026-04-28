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
STATE_SEARCH = 0
STATE_TRACK = 1
STATE_DONE = 2

state = STATE_SEARCH
active_target = None
reverse_mode = False

# -----------------------
# SAFE STOP
# -----------------------
def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)

# =========================================================
# 🔥 IMPROVED SEARCH MODE (FIX)
# =========================================================

search_angle = -25
search_dir = 1
last_search_switch = time.time()

def search_motion():
    """
    Continuous sweep + forward motion.
    Prevents freezing and improves re-acquisition.
    """
    global search_angle, search_dir, last_search_switch

    # slowly oscillate steering angle
    if time.time() - last_search_switch > 1.2:
        search_dir *= -1
        last_search_switch = time.time()

    search_angle += search_dir * 2
    search_angle = max(-30, min(30, search_angle))

    px.set_dir_servo_angle(search_angle)
    px.forward(25)

# -----------------------
# ACTIONS
# -----------------------
def AtMarker8():
    print("At Marker 8 → VEER")
    px.set_dir_servo_angle(35)
    px.forward(update_speed(speed))

def AtMarker10():
    print("At Marker 10 → TURN")
    px.set_dir_servo_angle(-45)
    px.forward(update_speed(speed))

# -----------------------
# TRACKING
# -----------------------
def track_marker_pnp(rvec, tvec, reverse=False):

    tvec = np.array(tvec).reshape(3,)
    x = float(tvec[0])
    z = float(tvec[2])

    if abs(z) < 1e-6:
        return

    steer = np.degrees(np.arctan2(x, z)) * 0.99

    if reverse:
        steer *= -1

    steer = float(np.clip(steer, -30, 30))

    px.set_dir_servo_angle(steer)
    px.forward(update_speed(speed))

    print(f"[TRACK] x:{x:.2f} z:{z:.2f} steer:{steer:.2f}")

    if z < 0.5:
        print("Close → trigger action")
        stop_car()
        return "close"

# -----------------------
# MAIN LOOP
# -----------------------
def main(headless=False):
    global state, active_target

    try:
        while state != STATE_DONE:

            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detector.detectMarkers(frame)

            # -----------------------
            # SEARCH MODE (FIXED)
            # -----------------------
            if state == STATE_SEARCH:

                search_motion()

                if ids is not None:
                    print("Marker found → TRACK")
                    state = STATE_TRACK
                    active_target = None

            # -----------------------
            # TRACK MODE
            # -----------------------
            elif state == STATE_TRACK:

                if ids is None or len(ids) == 0:
                    state = STATE_SEARCH
                    stop_car()
                    active_target = None
                    continue

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                marker_map = {int(id_val): i for i, id_val in enumerate(ids.flatten())}

                # target priority
                if active_target is None:
                    if 10 in marker_map:
                        active_target = 10
                    elif 8 in marker_map:
                        active_target = 8
                    elif 11 in marker_map:
                        active_target = 11

                if active_target not in marker_map:
                    continue

                i = marker_map[active_target]

                result = track_marker_pnp(
                    rvecs[i],
                    tvecs[i],
                    reverse_mode
                )

                # -----------------------
                # ACTIONS
                # -----------------------
                if result == "close":

                    if active_target == 8:
                        AtMarker8()
                        time.sleep(1.0)
                        stop_car()

                    elif active_target == 10:
                        AtMarker10()
                        time.sleep(2.0)
                        stop_car()

                    elif active_target == 11:
                        stop_car()

                    state = STATE_SEARCH
                    active_target = None
                    stop_car()
                    continue

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