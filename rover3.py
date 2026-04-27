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

marker_length = marker_size / 100  # cm → meters

# -----------------------
# Control params
# -----------------------
speed = 30
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

tracking = False
reverse_mode = False
active_target = None

# -----------------------
# STATES
# -----------------------
STATE_SEARCH = 0
STATE_TRACK = 1
STATE_DONE = 2

state = STATE_SEARCH

# -----------------------
# SAFE STOP
# -----------------------
def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)

# -----------------------
# SEARCH (NO PAN → ROTATE ROBOT)
# -----------------------
def search_motion():
    px.set_dir_servo_angle(25)
    px.forward(15)

# -----------------------
# SolvePnP tracking
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

    if reverse:
        px.backward(update_speed(speed))
    else:
        px.forward(update_speed(speed))

    if z < 0.5:
        print("Close to marker → stopping")
        stop_car()
        return "close"

    print(f"[{'REV' if reverse else 'FWD'}] x:{x:.2f} z:{z:.2f} steer:{steer:.2f}")

# -----------------------
# MAIN LOOP
# -----------------------
def main(headless=False):
    global state, tracking, reverse_mode, active_target

    try:
        while state != STATE_DONE:

            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detector.detectMarkers(frame)

            # -----------------------
            # SEARCH MODE (NO PAN)
            # -----------------------
            if state == STATE_SEARCH:

                search_motion()
                stop_car()

                if ids is not None:
                    print("Marker found → switching to TRACK")
                    state = STATE_TRACK
                    tracking = True
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
                    continue

                i = marker_map[active_target]

                result = track_marker_pnp(rvecs[i], tvecs[i], reverse_mode)

                # -----------------------
                # COMPLETION LOGIC
                # -----------------------
                if result == "close":

                    if active_target == 10:
                        print("Marker 10 → TURN LEFT")
                        stop_car()
                        px.set_dir_servo_angle(-25)
                        px.forward(update_speed(speed))
                        time.sleep(1.0)
                        stop_car()

                    elif active_target == 8:
                        print("Marker 8 → DONE")
                        stop_car()

                    elif active_target == 11:
                        print("Marker 11 → REVERSE DONE")
                        stop_car()

                    state = STATE_SEARCH
                    tracking = False
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

    except KeyboardInterrupt:
        pass

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