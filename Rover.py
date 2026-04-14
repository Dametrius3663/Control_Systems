import cv2
import numpy as np
from picarx import Picarxcd
import time

def generate_aruco_markers():
    """Generate ArUco markers for testing"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i in range(5):  # Generate markers 0-4
        marker = cv2.aruco.generateImageMarker(aruco_dict, i, 200)
        cv2.imwrite(f'aruco_marker_{i}.png', marker)
    print("ArUco markers generated: aruco_marker_0.png to aruco_marker_4.png")

def get_target_marker(state):
    if state in [STATE_SEARCH_1, STATE_APPROACH_1, STATE_RETRACE_SEARCH_1, STATE_RETRACE_APPROACH_1]:
        return 1
    elif state in [STATE_SEARCH_3, STATE_APPROACH_3, STATE_RETRACE_SEARCH_3, STATE_RETRACE_APPROACH_3]:
        return 3
    return None

def is_approach_state(state):
    return state in [STATE_APPROACH_1, STATE_APPROACH_3, STATE_RETRACE_APPROACH_1, STATE_RETRACE_APPROACH_3]

def is_search_state(state):
    return state in [STATE_SEARCH_1, STATE_SEARCH_3, STATE_RETRACE_SEARCH_1, STATE_RETRACE_SEARCH_3]

def is_turn_state(state):
    return state in [STATE_TURN_LEFT, STATE_TURN_180, STATE_RETRACE_TURN_RIGHT]

def next_state(current):
    transitions = {
        STATE_SEARCH_1: STATE_APPROACH_1,
        STATE_APPROACH_1: STATE_TURN_LEFT,
        STATE_TURN_LEFT: STATE_SEARCH_3,
        STATE_SEARCH_3: STATE_APPROACH_3,
        STATE_APPROACH_3: STATE_TURN_180,
        STATE_TURN_180: STATE_RETRACE_SEARCH_3,
        STATE_RETRACE_SEARCH_3: STATE_RETRACE_APPROACH_3,
        STATE_RETRACE_APPROACH_3: STATE_RETRACE_TURN_RIGHT,
        STATE_RETRACE_TURN_RIGHT: STATE_RETRACE_SEARCH_1,
        STATE_RETRACE_SEARCH_1: STATE_RETRACE_APPROACH_1,
        STATE_RETRACE_APPROACH_1: STATE_DONE
    }
    return transitions.get(current, STATE_DONE)

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
close_distance = 1.0  # 1 meter

# States
STATE_SEARCH_1 = 0
STATE_APPROACH_1 = 1
STATE_TURN_LEFT = 2
STATE_SEARCH_3 = 3
STATE_APPROACH_3 = 4
STATE_TURN_180 = 5
STATE_RETRACE_SEARCH_3 = 6
STATE_RETRACE_APPROACH_3 = 7
STATE_RETRACE_TURN_RIGHT = 8
STATE_RETRACE_SEARCH_1 = 9
STATE_RETRACE_APPROACH_1 = 10
STATE_DONE = 11

current_state = STATE_SEARCH_1
reverse_mode = False

def main():
    global current_state, reverse_mode
    try:
        while current_state != STATE_DONE:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(frame)

            target_id = get_target_marker(current_state)
            marker_detected = False
            distance = float('inf')
            target_corner = None

            if ids is not None and len(corners) > 0:
                # Estimate pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs)
                
                for i, id_val in enumerate(ids.flatten()):
                    if id_val == target_id:
                        marker_detected = True
                        distance = np.linalg.norm(tvecs[i][0])
                        target_corner = corners[i][0]
                        break

            # Check for obstacles
            front_distance = px.get_distance_at(0)  # front distance

            if front_distance < obstacle_threshold:
                # Obstacle detected
                px.stop()
                px.set_dir_servo_angle(turn_angle if not reverse_mode else -turn_angle)
                time.sleep(0.5)
                if reverse_mode:
                    px.backward(speed)
                else:
                    px.forward(speed)
                time.sleep(1)
                px.set_dir_servo_angle(0)
            else:
                if is_turn_state(current_state):
                    # Perform turn 
                    if current_state == STATE_TURN_LEFT:
                        px.set_dir_servo_angle(90)
                        time.sleep(1)
                    elif current_state == STATE_TURN_180:
                        px.set_dir_servo_angle(180)
                        time.sleep(2)
                        reverse_mode = True
                    elif current_state == STATE_RETRACE_TURN_RIGHT:
                        px.set_dir_servo_angle(-90)
                        time.sleep(1)
                    px.set_dir_servo_angle(0)
                    current_state = next_state(current_state)
                elif is_approach_state(current_state):
                    if marker_detected and distance < close_distance:
                        # Reached target
                        px.stop()
                        current_state = next_state(current_state)
                    elif marker_detected:
                        # Steer towards marker
                        center_x = (target_corner[0][0] + target_corner[1][0] + target_corner[2][0] + target_corner[3][0]) / 4
                        frame_center_x = frame.shape[1] / 2
                        error = center_x - frame_center_x
                        k_p = 0.1
                        steer_angle = -error * k_p
                        steer_angle = np.clip(steer_angle, -45, 45)
                        px.set_dir_servo_angle(steer_angle)
                        if reverse_mode:
                            px.backward(speed)
                        else:
                            px.forward(speed)
                    else:
                        # Marker lost, search by turning
                        px.set_dir_servo_angle(turn_angle if not reverse_mode else -turn_angle)
                        if reverse_mode:
                            px.backward(speed)
                        else:
                            px.forward(speed)
                        time.sleep(0.5)
                        px.set_dir_servo_angle(0)
                elif is_search_state(current_state):
                    if marker_detected:
                        current_state = next_state(current_state)
                    else:
                        # Search by turning
                        px.set_dir_servo_angle(turn_angle if not reverse_mode else -turn_angle)
                        if reverse_mode:
                            px.backward(speed)
                        else:
                            px.forward(speed)
                        time.sleep(0.5)
                        px.set_dir_servo_angle(0)

            # Display frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow('ArUco Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

            print(f"State: {current_state}, Target: {target_id}, Distance: {distance:.2f}, Reverse: {reverse_mode}")

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        px.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    main()
