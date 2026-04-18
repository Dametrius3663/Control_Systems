import cv2
import numpy as np
from picarx import Picarxcd
import threading
import time

# Hardware
px = Picarxcd()

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Camera parameters for pose estimation (approximate)
marker_length = 0.05  # 5cm markers
camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))

# Movement parameters
speed = 30
turn_angle = 30
obstacle_threshold = 30  # cm
marker_close_area = 0.02  # marker covers this fraction of frame area when close enough

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
reverse_mode = False

# Obstacle avoidance background monitor
obstacle_data = {
    "distance_cm": None,
    "detected": False,
    "last_check": 0.0,
}
obstacle_lock = threading.Lock()
avoidance_shutdown = threading.Event()


def get_front_distance():
    if hasattr(px, "ultrasonic"):
        try:
            return float(px.ultrasonic.read())
        except Exception:
            pass
    if hasattr(px, "get_distance_at"):
        try:
            return float(px.get_distance_at(0))
        except Exception:
            pass
    return None


def object_avoidance_loop():
    while not avoidance_shutdown.is_set():
        dist = get_front_distance()
        with obstacle_lock:
            if dist is not None:
                obstacle_data["distance_cm"] = dist
                obstacle_data["detected"] = dist < obstacle_threshold
                obstacle_data["last_check"] = time.time()
            else:
                obstacle_data["detected"] = False
        time.sleep(0.1)


def perform_obstacle_avoidance():
    stop_car()
    px.set_dir_servo_angle(0)
    # Back up briefly before turning
    px.backward(speed)
    time.sleep(0.5)
    px.stop()

    # Turn away from the obstacle
    turn_dir = -turn_angle if reverse_mode else turn_angle
    px.set_dir_servo_angle(turn_dir)
    px.forward(speed)
    time.sleep(0.8)
    px.stop()
    px.set_dir_servo_angle(0)


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


def steer_from_pixel(center_x, frame_width, k_p=0.15, max_angle=45):
    error = center_x - frame_width / 2
    steer_angle = -error * k_p
    return float(np.clip(steer_angle, -max_angle, max_angle))


def turn_right():
    px.stop()
    px.set_dir_servo_angle(-90)
    time.sleep(1)
    px.set_dir_servo_angle(0)


def turn_left():
    px.stop()
    px.set_dir_servo_angle(90)
    time.sleep(1)
    px.set_dir_servo_angle(0)


def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)


def drive_forward(steer_angle=0):
    px.set_dir_servo_angle(steer_angle)
    px.forward(speed)


def search_motion():
    px.set_dir_servo_angle(turn_angle if not reverse_mode else -turn_angle)
    px.forward(speed)
    time.sleep(0.5)
    px.set_dir_servo_angle(0)


def turn_around():
    px.stop()
    px.set_dir_servo_angle(180)
    time.sleep(2)
    px.set_dir_servo_angle(0)


def main():
    global current_state, reverse_mode
    try:
        avoidance_thread = threading.Thread(target=object_avoidance_loop, daemon=True)
        avoidance_thread.start()

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

            with obstacle_lock:
                obstacle_detected = obstacle_data["detected"]
                obstacle_distance = obstacle_data["distance_cm"]

            if obstacle_detected:
                print(f"Obstacle detected at {obstacle_distance:.1f} cm, avoiding")
                perform_obstacle_avoidance()
                continue

            marker1 = marker_data.get(1)
            marker2 = marker_data.get(2)
            marker3 = marker_data.get(3)
            marker4 = marker_data.get(4)
            marker5 = marker_data.get(5)

            if current_state == STATE_SEARCH_1:
                if marker1:
                    current_state = STATE_APPROACH_1
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_1:
                if marker_is_close(marker1):
                    stop_car()
                    current_state = STATE_SEARCH_2
                elif marker1:
                    steer = steer_from_pixel(marker1["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_1

            elif current_state == STATE_SEARCH_2:
                if marker2:
                    current_state = STATE_APPROACH_2
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_2:
                if marker_is_close(marker2):
                    stop_car()
                    current_state = STATE_SEARCH_3
                elif marker2:
                    steer = steer_from_pixel(marker2["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_2

            elif current_state == STATE_SEARCH_3:
                if marker3:
                    current_state = STATE_APPROACH_3
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_3:
                if marker_is_close(marker3):
                    stop_car()
                    current_state = STATE_SEARCH_4
                elif marker3:
                    steer = steer_from_pixel(marker3["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_3

            elif current_state == STATE_SEARCH_4:
                if marker4:
                    current_state = STATE_APPROACH_4
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_4:
                if marker_is_close(marker4):
                    stop_car()
                    current_state = STATE_SEARCH_5
                elif marker4:
                    steer = steer_from_pixel(marker4["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_4

            elif current_state == STATE_SEARCH_5:
                if marker5:
                    current_state = STATE_APPROACH_5
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_5:
                if marker_is_close(marker5):
                    stop_car()
                    current_state = STATE_TURN_AROUND
                elif marker5:
                    steer = steer_from_pixel(marker5["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_5

            elif current_state == STATE_TURN_AROUND:
                turn_around()
                reverse_mode = True
                current_state = STATE_REVERSE_SEARCH_4

            elif current_state == STATE_REVERSE_SEARCH_4:
                if marker4:
                    current_state = STATE_REVERSE_APPROACH_4
                else:
                    search_motion()

            elif current_state == STATE_REVERSE_APPROACH_4:
                if marker_is_close(marker4):
                    stop_car()
                    current_state = STATE_REVERSE_SEARCH_3
                elif marker4:
                    steer = steer_from_pixel(marker4["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_REVERSE_SEARCH_4

            elif current_state == STATE_REVERSE_SEARCH_3:
                if marker3:
                    current_state = STATE_REVERSE_APPROACH_3
                else:
                    search_motion()

            elif current_state == STATE_REVERSE_APPROACH_3:
                if marker_is_close(marker3):
                    stop_car()
                    current_state = STATE_REVERSE_SEARCH_2
                elif marker3:
                    steer = steer_from_pixel(marker3["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_REVERSE_SEARCH_3

            elif current_state == STATE_REVERSE_SEARCH_2:
                if marker2:
                    current_state = STATE_REVERSE_APPROACH_2
                else:
                    search_motion()

            elif current_state == STATE_REVERSE_APPROACH_2:
                if marker_is_close(marker2):
                    stop_car()
                    current_state = STATE_REVERSE_SEARCH_1
                elif marker2:
                    steer = steer_from_pixel(marker2["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_REVERSE_SEARCH_2

            elif current_state == STATE_REVERSE_SEARCH_1:
                if marker1:
                    current_state = STATE_REVERSE_APPROACH_1
                else:
                    search_motion()

            elif current_state == STATE_REVERSE_APPROACH_1:
                if marker_is_close(marker1):
                    stop_car()
                    current_state = STATE_DONE
                elif marker1:
                    steer = steer_from_pixel(marker1["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_REVERSE_SEARCH_1

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow('ArUco Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

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
        avoidance_shutdown.set()
        stop_car()
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
