import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
from picarx import Picarx
from Vision.app.core.config import Config

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

marker_length = marker_size / 100  # cm → m

# -----------------------
# Control params
# -----------------------
speed = 30
reverse_mode = False

# -----------------------
# State machine
# -----------------------
STATE_SEARCH_1 = 0
STATE_APPROACH_1 = 1
STATE_SEARCH_2 = 2
STATE_APPROACH_2 = 3
STATE_SEARCH_3 = 4
STATE_APPROACH_3 = 5
STATE_SEARCH_4 = 6
STATE_APPROACH_4 = 7
STATE_SEARCH_5 = 8
STATE_APPROACH_5 = 9
STATE_TURN_AROUND = 10
STATE_REVERSE_SEARCH_4 = 11
STATE_REVERSE_APPROACH_4 = 12
STATE_REVERSE_SEARCH_3 = 13
STATE_REVERSE_APPROACH_3 = 14
STATE_REVERSE_SEARCH_2 = 15
STATE_REVERSE_APPROACH_2 = 16
STATE_REVERSE_SEARCH_1 = 17
STATE_REVERSE_APPROACH_1 = 18
STATE_DONE = 19

current_state = STATE_SEARCH_1


# -----------------------
# SolvePnP-based tracking
# -----------------------
def track_marker_pnp(rvec, tvec, reverse=False):
    """
    Use SolvePnP output for stable steering.
    """

    x = tvec[0][0][0]
    z = tvec[0][0][2]

    # prevent divide-by-zero
    if z < 1e-6:
        return

    # yaw angle (radians)
    yaw = np.arctan2(x, z)

    # convert to steering command
    steer = np.degrees(yaw) * -1.2 # gain

    if reverse:
        steer = -steer

    steer = float(np.clip(steer, -30, 30))

    px.set_dir_servo_angle(steer)

    if reverse:
        px.backward(speed)
    else:
        px.forward(speed)

    print(f"[{'REV' if reverse else 'FWD'}] x:{x:.2f} z:{z:.2f} yaw:{np.degrees(yaw):.2f} steer:{steer:.2f}")


def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)


# -----------------------
# Main loop
# -----------------------
def main(headless=False):
    global current_state, reverse_mode

    try:
        while current_state != STATE_DONE:

            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, rejected = detector.detectMarkers(frame)

            marker_data = {}

            if ids is not None and len(corners) > 0:

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                for i, id_val in enumerate(ids.flatten()):
                    marker_data[int(id_val)] = (rvecs[i], tvecs[i])

            # -----------------------
            # Marker mapping
            # -----------------------
            m1 = marker_data.get(8)
            m2 = marker_data.get(9)
            m3 = marker_data.get(10)
            m4 = marker_data.get(11)
            m5 = marker_data.get(12)

            # -----------------------
            # STATE MACHINE (simplified logic)
            # -----------------------

            if current_state == STATE_SEARCH_1:
                if m1:
                    current_state = STATE_APPROACH_1

            elif current_state == STATE_APPROACH_1:
                if m1:
                    track_marker_pnp(m1[0], m1[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_SEARCH_2:
                if m2:
                    current_state = STATE_APPROACH_2

            elif current_state == STATE_APPROACH_2:
                if m2:
                    track_marker_pnp(m2[0], m2[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_SEARCH_3:
                if m3:
                    current_state = STATE_APPROACH_3

            elif current_state == STATE_APPROACH_3:
                if m3:
                    track_marker_pnp(m3[0], m3[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_SEARCH_4:
                if m4:
                    current_state = STATE_APPROACH_4

            elif current_state == STATE_APPROACH_4:
                if m4:
                    track_marker_pnp(m4[0], m4[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_SEARCH_5:
                if m5:
                    current_state = STATE_APPROACH_5

            elif current_state == STATE_APPROACH_5:
                if m5:
                    track_marker_pnp(m5[0], m5[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_TURN_AROUND:
                stop_car()
                reverse_mode = True
                current_state = STATE_REVERSE_SEARCH_4

            elif current_state == STATE_REVERSE_SEARCH_4:
                if m4:
                    current_state = STATE_REVERSE_APPROACH_4

            elif current_state == STATE_REVERSE_APPROACH_4:
                if m4:
                    track_marker_pnp(m4[0], m4[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_REVERSE_SEARCH_3:
                if m3:
                    current_state = STATE_REVERSE_APPROACH_3

            elif current_state == STATE_REVERSE_APPROACH_3:
                if m3:
                    track_marker_pnp(m3[0], m3[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_REVERSE_SEARCH_2:
                if m2:
                    current_state = STATE_REVERSE_APPROACH_2

            elif current_state == STATE_REVERSE_APPROACH_2:
                if m2:
                    track_marker_pnp(m2[0], m2[1], reverse_mode)
                else:
                    stop_car()

            elif current_state == STATE_REVERSE_SEARCH_1:
                if m1:
                    current_state = STATE_REVERSE_APPROACH_1

            elif current_state == STATE_REVERSE_APPROACH_1:
                if m1:
                    track_marker_pnp(m1[0], m1[1], reverse_mode)
                else:
                    stop_car()

            # -----------------------
            # display
            # -----------------------
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if not headless:
                cv2.imshow("ArUco", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    finally:
        stop_car()
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(args.headless)