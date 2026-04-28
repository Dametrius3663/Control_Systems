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
# STATE MACHINE
# -----------------------
STATE_SEARCH = 0
STATE_TRACK = 1
STATE_DONE = 2

state = STATE_SEARCH
tracking = False
active_target = None
reverse_mode = False

# -----------------------
# LOSS PERSISTENCE (FIX)
# -----------------------
last_seen_time = 0
lost_timeout = 0.6  # seconds before declaring marker "gone"

# -----------------------
# SAFE STOP
# -----------------------
def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)

# -----------------------
# SEARCH MODE (stable, no jitter)
# -----------------------
def search_motion():
    # slow forward crawl with slight steering bias
    px.set_dir_servo_angle(20)
    px.forward(10)

# -----------------------
# TRACKING FUNCTION WITH CONFIDENCE
# -----------------------
def track_marker_pnp(rvec, tvec, confidence, reverse=False):

    tvec = np.array(tvec).reshape(3,)
    x = float(tvec[0])
    z = float(tvec[2])

    if abs(z) < 1e-6:
        return

    # basic angle control
    steer = np.degrees(np.arctan2(x, z)) * 0.99

    if reverse:
        steer *= -1

    # confidence weighting (stabilizes far detection)
    steer *= confidence

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

    print(f"[TRACK] x:{x:.2f} z:{z:.2f} conf:{confidence:.2f} steer:{steer:.2f}")

# -----------------------
# MAIN LOOP
# -----------------------
def main(headless=False):
    global state, tracking, active_target, last_seen_time

    try:
        while state != STATE_DONE:

            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detector.detectMarkers(frame)

            # =======================
            # MARKER DETECTION STABILITY FIX
            # =======================

            marker_visible = ids is not None and len(ids) > 0

            if marker_visible:
                last_seen_time = time.time()

            # =======================
            # SEARCH MODE
            # =======================
            if state == STATE_SEARCH:

                search_motion()
                stop_car()

                if marker_visible:
                    print("Marker detected → switching to TRACK")
                    state = STATE_TRACK
                    tracking = True
                    active_target = None

            # =======================
            # TRACK MODE
            # =======================
            elif state == STATE_TRACK:

                # -----------------------
                # LOST MARKER HANDLING (FIX)
                # -----------------------
                if not marker_visible:
                    if time.time() - last_seen_time > lost_timeout:
                        print("Marker lost → returning to SEARCH")
                        state = STATE_SEARCH
                        tracking = False
                        active_target = None
                        stop_car()
                        continue
                    else:
                        # short dropout → ignore (prevents shake)
                        px.forward(update_speed(40))
                        continue

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                marker_map = {int(id_val): i for i, id_val in enumerate(ids.flatten())}

                # -----------------------
                # TARGET SELECTION
                # -----------------------
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

                # -----------------------
                # CONFIDENCE FROM AREA (STABILITY FIX)
                # -----------------------
                area = cv2.contourArea(corners[i][0])
                confidence = np.clip(area / 5000.0, 0.2, 1.0)

                result = track_marker_pnp(
                    rvecs[i],
                    tvecs[i],
                    confidence,
                    reverse_mode
                )

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

            # =======================
            # DISPLAY
            # =======================
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