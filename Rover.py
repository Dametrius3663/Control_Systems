import cv2
import numpy as np
from picarx import Picarxcd
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
marker1_threshold = 0.6  # 24 inches
marker4_threshold = 0.61  # 24 inches
marker5_threshold = 0.91  # 36 inches

# States
STATE_SEARCH_1 = 0
STATE_APPROACH_1 = 1
STATE_SEARCH_2_3 = 2
STATE_DRIVE_BETWEEN_2_3 = 3
STATE_SEARCH_4 = 4
STATE_APPROACH_4 = 5
STATE_TURN_LEFT_TO_5 = 6
STATE_SEARCH_5 = 7
STATE_APPROACH_5 = 8
STATE_TURN_AROUND = 9
STATE_REVERSE_SEARCH_4 = 10
STATE_REVERSE_APPROACH_4 = 11
STATE_REVERSE_SEARCH_2_3 = 12
STATE_REVERSE_DRIVE_BETWEEN_2_3 = 13
STATE_REVERSE_SEARCH_1 = 14
STATE_REVERSE_APPROACH_1 = 15
STATE_DONE = 16

current_state = STATE_SEARCH_1
reverse_mode = False


def get_marker_data(corners, ids, tvecs):
    data = {}
    if ids is None:
        return data
    for i, id_val in enumerate(ids.flatten()):
        center = np.mean(corners[i][0], axis=0)
        distance = float(np.linalg.norm(tvecs[i][0]))
        data[int(id_val)] = {
            "center": center,
            "distance": distance,
            "corners": corners[i][0],
        }
    return data


def steer_from_pixel(center_x, frame_width, k_p=0.15, max_angle=45):
    error = center_x - frame_width / 2
    steer_angle = -error * k_p
    return float(np.clip(steer_angle, -max_angle, max_angle))


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


def turn_left():
    px.stop()
    px.set_dir_servo_angle(90)
    time.sleep(1)
    px.set_dir_servo_angle(0)


def turn_around():
    px.stop()
    px.set_dir_servo_angle(180)
    time.sleep(2)
    px.set_dir_servo_angle(0)


def main():
    global current_state, reverse_mode
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
                marker_data = get_marker_data(corners, ids, tvecs)

            marker1 = marker_data.get(1)
            marker2 = marker_data.get(2)
            marker3 = marker_data.get(3)
            marker4 = marker_data.get(4)
            marker5 = marker_data.get(5)

            front_distance = px.get_distance_at(0)

            if front_distance < obstacle_threshold:
                stop_car()
                px.set_dir_servo_angle(turn_angle if not reverse_mode else -turn_angle)
                time.sleep(0.5)
                px.forward(speed)
                time.sleep(1)
                px.set_dir_servo_angle(0)
                continue

            if current_state == STATE_SEARCH_1:
                if marker1:
                    current_state = STATE_APPROACH_1
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_1:
                if marker1 and marker1["distance"] < marker1_threshold:
                    stop_car()
                    current_state = STATE_SEARCH_2_3
                elif marker1:
                    steer = steer_from_pixel(marker1["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_1

            elif current_state == STATE_SEARCH_2_3:
                if marker2 and marker3:
                    current_state = STATE_DRIVE_BETWEEN_2_3
                else:
                    search_motion()

            elif current_state == STATE_DRIVE_BETWEEN_2_3:
                if marker4:
                    stop_car()
                    current_state = STATE_SEARCH_4
                elif marker2 and marker3:
                    midpoint = (marker2["center"][0] + marker3["center"][0]) / 2
                    steer = steer_from_pixel(midpoint, frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_2_3

            elif current_state == STATE_SEARCH_4:
                if marker4:
                    current_state = STATE_APPROACH_4
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_4:
                if marker4 and marker4["distance"] < marker4_threshold:
                    stop_car()
                    current_state = STATE_TURN_LEFT_TO_5
                elif marker4:
                    steer = steer_from_pixel(marker4["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_SEARCH_4

            elif current_state == STATE_TURN_LEFT_TO_5:
                turn_left()
                current_state = STATE_SEARCH_5

            elif current_state == STATE_SEARCH_5:
                if marker5:
                    current_state = STATE_APPROACH_5
                else:
                    search_motion()

            elif current_state == STATE_APPROACH_5:
                if marker5 and marker5["distance"] < marker5_threshold:
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
                if marker4 and marker4["distance"] < marker4_threshold:
                    stop_car()
                    current_state = STATE_REVERSE_SEARCH_2_3
                elif marker4:
                    steer = steer_from_pixel(marker4["center"][0], frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_REVERSE_SEARCH_4

            elif current_state == STATE_REVERSE_SEARCH_2_3:
                if marker2 and marker3:
                    current_state = STATE_REVERSE_DRIVE_BETWEEN_2_3
                else:
                    search_motion()

            elif current_state == STATE_REVERSE_DRIVE_BETWEEN_2_3:
                if marker1:
                    stop_car()
                    current_state = STATE_REVERSE_APPROACH_1
                elif marker2 and marker3:
                    midpoint = (marker2["center"][0] + marker3["center"][0]) / 2
                    steer = steer_from_pixel(midpoint, frame.shape[1])
                    drive_forward(steer)
                else:
                    current_state = STATE_REVERSE_SEARCH_2_3

            elif current_state == STATE_REVERSE_APPROACH_1:
                if marker1 and marker1["distance"] < marker1_threshold:
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
        stop_car()
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
