import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
from picarx import Picarx
from Vision.app.core.config import Config

# Hardware
px = Picarx()
config = Config.get_instance() 
# --- ArUco Setup ---
aruco_dict = config.aruco_dict
aruco_params = config.aruco_params

cam_matrix = config.cam_matrix
dist_coeffs = config.dist_coeffs
marker_size = config.marker_size
# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Camera parameters for pose estimation (approximate)
marker_length = marker_size / 100  # Convert cm to meters
camera_matrix = cam_matrix
dist_coeffs = dist_coeffs

# Pan parameters
pan_start = -45  # degrees
pan_end = 45
pan_increment = 1
pan_delay = 0.5  # seconds between steps
marker_close_area = 0.15  # 15% of frame = close enough
headless = False  # Default to showing display

reverse_mode = False

# States
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
current_pan = 0  # Track current pan position for non-blocking pan
speed = 30

def get_marker_data(corners, ids, tvecs, frame_area):
    data = {}
    if ids is None:
        return data
    for i, id_val in enumerate(ids.flatten()):
        center = np.mean(corners[i][0], axis=0)
        distance = float(np.linalg.norm(tvecs[i][0]))
        xs = corners[i][0][:, 0]
        ys = corners[i][0][:, 1]
        bbox_area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0
        data[int(id_val)] = {
            "center": center,
            "distance": distance,
            "corners": corners[i][0],
            "area_ratio": area_ratio,
        }
    return data


def marker_is_close(marker, threshold=marker_close_area):
    return marker is not None and marker.get("area_ratio", 0.0) >= threshold

def pan_step():
    """Take one step in pan sweep (non-blocking)."""
    global current_pan
    if current_pan > pan_start:
        current_pan -= pan_increment
        px.set_cam_pan_angle(current_pan)
    else:
        # Reset to start when pan complete
        current_pan = pan_end


def reset_pan():
    """Reset pan to starting position."""
    global current_pan
    current_pan = 0
    px.set_cam_pan_angle(0)

def track_marker(marker, frame_width, current_pan=0, reverse=False):
    """Track marker using camera POV for direction."""

    if marker:
        marker_x = marker["center"][0]
        frame_center = frame_width / 2
        error = marker_x - frame_center

        # --- PAN CAMERA (for visibility only) ---
        new_pan = np.clip(error * 0.02, -45, 45)
        new_pan = float(np.clip(new_pan, -45, 45)) 
        px.set_cam_pan_angle(new_pan)

        # --- STEERING BASED ON CAMERA POV ---
        steer = error * 0.03

        # Flip steering if reversing
        if reverse:
            steer = -steer

        steer = float(np.clip(steer, -30, 30))
        px.set_dir_servo_angle(steer)

        # --- MOTION ---
        if reverse:
            px.backward(speed)
        else:
            px.forward(speed)

        print(f"[{'REV' if reverse else 'FWD'}] x:{marker_x:.0f} err:{error:.0f} pan:{new_pan:.1f} steer:{steer:.1f}")

        return new_pan

    return current_pan


def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)

def main(headless=False):
    global current_state, reverse_mode, current_pan
    try:
        while current_state != STATE_DONE:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame")
                break

            corners, ids, rejected = detector.detectMarkers(frame)
            marker_data = {}

            if ids is not None and len(corners) > 0:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs)
                frame_area = frame.shape[0] * frame.shape[1]
                marker_data = get_marker_data(corners, ids, tvecs, frame_area)

            marker1 = marker_data.get(8)
            marker2 = marker_data.get(9)
            marker3 = marker_data.get(10)
            marker4 = marker_data.get(11)
            marker5 = marker_data.get(12)

            if current_state == STATE_SEARCH_1:
                if marker1:
                    current_state = STATE_APPROACH_1
                else:
                    pan_step()

            elif current_state == STATE_APPROACH_1:
                if marker_is_close(marker1):
                    stop_car()
                    reset_pan()
                    current_state = STATE_SEARCH_2
                else:
                    current_pan = track_marker(marker1, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_SEARCH_2:
                if marker2:
                    current_state = STATE_APPROACH_2
                else:
                    pan_step()

            elif current_state == STATE_APPROACH_2:
                if marker_is_close(marker2):
                    stop_car()
                    reset_pan()
                    current_state = STATE_SEARCH_3
                else:
                    current_pan = track_marker(marker2, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_SEARCH_3:
                if marker3:
                    current_state = STATE_APPROACH_3
                else:
                    pan_step()

            elif current_state == STATE_APPROACH_3:
                if marker_is_close(marker3):
                    stop_car()
                    reset_pan()
                    current_state = STATE_SEARCH_4
                else:
                    current_pan = track_marker(marker3, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_SEARCH_4:
                if marker4:
                    current_state = STATE_APPROACH_4
                else:
                    pan_step()

            elif current_state == STATE_APPROACH_4:
                if marker_is_close(marker4):
                    stop_car()
                    reset_pan()
                    current_state = STATE_SEARCH_5
                else:
                    current_pan = track_marker(marker4, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_SEARCH_5:
                if marker5:
                    current_state = STATE_APPROACH_5
                else:
                    pan_step()

            elif current_state == STATE_APPROACH_5:
                if marker_is_close(marker5):
                    stop_car()
                    reset_pan()
                    current_state = STATE_TURN_AROUND
                else:
                    current_pan = track_marker(marker5, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_TURN_AROUND:
                stop_car()
                reset_pan()
                reverse_mode = True
                current_state = STATE_REVERSE_SEARCH_4

            elif current_state == STATE_REVERSE_SEARCH_4:
                if marker4:
                    current_state = STATE_REVERSE_APPROACH_4
                else:
                    pan_step()

            elif current_state == STATE_REVERSE_APPROACH_4:
                if marker_is_close(marker4):
                    stop_car()
                    reset_pan()
                    current_state = STATE_REVERSE_SEARCH_3
                else:
                    current_pan = track_marker(marker4, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_REVERSE_SEARCH_3:
                if marker3:
                    current_state = STATE_REVERSE_APPROACH_3
                else:
                    pan_step()

            elif current_state == STATE_REVERSE_APPROACH_3:
                if marker_is_close(marker3):
                    stop_car()
                    reset_pan()
                    current_state = STATE_REVERSE_SEARCH_2
                else:
                    current_pan = track_marker(marker3, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_REVERSE_SEARCH_2:
                if marker2:
                    current_state = STATE_REVERSE_APPROACH_2
                else:
                    pan_step()

            elif current_state == STATE_REVERSE_APPROACH_2:
                if marker_is_close(marker2):
                    stop_car()
                    reset_pan()
                    current_state = STATE_REVERSE_SEARCH_1
                else:
                    current_pan = track_marker(marker2, frame.shape[1], current_pan, reverse_mode)

            elif current_state == STATE_REVERSE_SEARCH_1:
                if marker1:
                    current_state = STATE_REVERSE_APPROACH_1
                else:
                    pan_step()

            elif current_state == STATE_REVERSE_APPROACH_1:
                if marker_is_close(marker1):
                    stop_car()
                    current_state = STATE_DONE
                else:
                    current_pan = track_marker(marker1, frame.shape[1], current_pan, reverse_mode)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if not headless:
                try:
                    cv2.imshow('ArUco Detection', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                except Exception as e:
                    print(f"Display error: {e}, continuing in headless mode")
                    headless = True

            print(
                f"State: {current_state}, "
                f"1:{'Y' if marker1 else 'N'} "
                f"2:{'Y' if marker2 else 'N'} "
                f"3:{'Y' if marker3 else 'N'} "
                f"4:{'Y' if marker4 else 'N'} "
                f"5:{'Y' if marker5 else 'N'} "
                f"Reverse:{reverse_mode}"
            )

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        stop_car()
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rover with optional headless mode.")
    parser.add_argument("--headless", action="store_true", help="Disable GUI display and run without opening windows.")
    args = parser.parse_args()
    main(headless=args.headless)
