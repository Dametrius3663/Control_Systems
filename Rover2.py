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
speed = 10

pan_start = -45
pan_end = 45
pan_increment = 1
current_pan = 0
pan_dir = 1

tracking = False
reverse_mode = False

# -----------------------
# STATES
# -----------------------
STATE_SEARCH = 0
STATE_TRACK = 1
STATE_DONE = 2

state = STATE_SEARCH

# -----------------------
# SAFE STOP (FIXED ERROR)
# -----------------------
def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)

# -----------------------
# PAN SWEEP (SEARCH MODE)
# -----------------------
def pan_sweep():
    global current_pan, pan_dir
    current_pan += pan_dir * pan_increment
    if current_pan >= pan_end:
        pan_dir = -1
    elif current_pan <= pan_start:
        pan_dir = 1
    px.set_cam_pan_angle(current_pan)
# -----------------------
# SolvePnP tracking
# -----------------------
def track_marker_pnp(rvec, tvec, reverse=False):

    tvec = np.array(tvec).reshape(3,)

    x = float(tvec[0])
    z = float(tvec[2])

    if abs(z) < 1e-6:
        return

    yaw = np.arctan2(x, z)

    steer = np.degrees(yaw) * 0.99

    if reverse:
        steer *= -1

    steer = float(np.clip(steer, -30, 30))

    px.set_dir_servo_angle(steer)

    if reverse:
        px.backward(speed)
    else:
        px.forward(speed)
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
            # SEARCH MODE
            # -----------------------
            if state == STATE_SEARCH:

                pan_sweep()
                stop_car()

                if ids is not None:
                    print("Marker found → switching to TRACK")
                    state = STATE_TRACK
                    tracking = True
                    active_target = None  # reset latch when entering TRACK

            # -----------------------
            # TRACK MODE
            # -----------------------
            elif state == STATE_TRACK:

                # SAFETY CHECK
                if ids is None or len(ids) == 0:
                    state = STATE_SEARCH
                    stop_car()
                    active_target = None
                    continue

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                marker_map = {int(id_val): i for i, id_val in enumerate(ids.flatten())}

                # ----------------------------
                # LATCH TARGET (STABLE PRIORITY)
                # ----------------------------
                if active_target is None:
                    if 10 in marker_map:
                        active_target = 10
                    elif 8 in marker_map:
                        active_target = 8
                    elif 11 in marker_map:
                        active_target = 11

                # If target disappears → reset latch
                if active_target not in marker_map:
                    active_target = None
                    continue

                i = marker_map[active_target]

                # ----------------------------
                # EXECUTE BEHAVIOR
                # ----------------------------
                result = track_marker_pnp(
                    rvecs[i],
                    tvecs[i],
                    reverse_mode
                )
                # ----------------------------
                # COMPLETION LOGIC
                # ----------------------------
                if result == "close":
                    if active_target == 10:
                        print("Marker 10 → CLOSE → TURN LEFT")
                        stop_car()
                        px.set_dir_servo_angle(-25)
                        px.forward(speed)
                        time.sleep(1.0)
                        stop_car()
                    elif active_target == 8:
                        print("Marker 8 → DONE")
                        stop_car()
                    elif active_target == 11:
                        print("Marker 11 → REVERSE DONE")
                        stop_car()
                    # reset system after any action
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