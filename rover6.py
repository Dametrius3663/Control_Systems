import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
from picarx import Picarx
from Vision.app.core.config import Config
import time
import os
from datetime import datetime
import pyttsx3

# Photo capture setup
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)
last_capture_time = 0
capture_interval = 30  # seconds

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # speed of speech

# Hardware
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

# Control
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

close_counter = 0
close_threshold_frames = 5

lost_counter = 0
lost_threshold_frames = 10

# -----------------------
# SAFE STOP
# -----------------------
def stop_car():
    px.stop()
    px.set_dir_servo_angle(0)

# -----------------------
# ACTIONS
# -----------------------
def AtMarker1():
    print("Marker 1 → VEER")
    px.set_dir_servo_angle(20)
    px.forward(update_speed(speed))

def AtMarker2():
    print("Marker 2 → COMPLEX PATH")
    px.set_dir_servo_angle(15)
    px.forward(update_speed(speed))
    time.sleep(0.5)
    px.set_dir_servo_angle(-4)
    px.forward(update_speed(speed))
    time.sleep(3.75)
    px.set_dir_servo_angle(25)
    px.forward(update_speed(speed))
    time.sleep(0.75)

def AtMarker4():
    print("Marker 4 → VEER PATH")
    px.set_dir_servo_angle(-25)
    px.forward(update_speed(speed))
    time.sleep(1)
    px.set_dir_servo_angle(-4)
    time.sleep(3.2)

def AtMarker6():
    print("Marker 6 → VEER")
    px.set_dir_servo_angle(-45)
    px.backward(update_speed(speed))

def AtMarker10():
    print("Marker 10 → TURN")
    px.set_dir_servo_angle(-45)
    px.forward(update_speed(speed))
    time.sleep(2)
    px.set_dir_servo_angle(-4)
    px.forward(update_speed(speed))
    time.sleep(4.5)
    px.set_dir_servo_angle(-20)
    px.forward(update_speed(speed))
    time.sleep(1.2)

def AtMarker11():
    print("Marker 11 → STRAIGHT")
    px.set_dir_servo_angle(-10)
    px.forward(update_speed(speed))

def AtMarker12():
    print("Marker 12 → REVERSE")
    px.set_dir_servo_angle(-45)
    px.backward(update_speed(speed))

def AtMarker15():
    print("Marker 15 → VEER")
    px.set_dir_servo_angle(25)
    px.forward(update_speed(speed))

def AtMarker17():
    print("Marker 17 → VEER")
    px.set_dir_servo_angle(25)
    px.forward(update_speed(speed))

#Speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# TRACKING
def track_marker_pnp(rvec, tvec, reverse=False):
    global active_target, close_counter
    tvec = np.array(tvec).reshape(3,)
    x = float(tvec[0])
    z = float(tvec[2])
    if abs(z) < 1e-6:
        close_counter = 0
        return
    steer = (np.degrees(np.arctan2(x, z)) * 0.99)-4
    if reverse:
        steer *= -1
    steer = float(np.clip(steer, -30, 30))
    px.set_dir_servo_angle(steer)
    px.forward(update_speed(speed))
    print(f"[TRACK] id:{active_target} x:{x:.2f} z:{z:.2f} steer:{steer:.2f} close:{close_counter}")
    # Frame lock logic
    if z < 1.75:
        close_counter += 1
    else:
        close_counter = 0
    if close_counter >= close_threshold_frames:
        print("LOCKED CLOSE → executing action")
        stop_car()
        target = active_target
        if target == 1:
            AtMarker1()
            time.sleep(0.1)
            stop_car()
        elif target == 2:
            AtMarker2()
            stop_car()
        elif target == 4:
            AtMarker4()
            stop_car()
        elif target == 6:
            AtMarker6()
            time.sleep(2.9)
            stop_car()
        elif target == 10:
            AtMarker10()
            stop_car()
        elif target == 11:
            AtMarker11()
            time.sleep(0.15)
            px.set_dir_servo_angle(10)
            px.forward(update_speed(speed))
            time.sleep(0.075)
            stop_car()
        elif target == 12:
            AtMarker12()
            time.sleep(2.9)
            stop_car()
        elif target == 15:
            AtMarker15()
            time.sleep(1.5)
            px.forward(update_speed(speed))
            time.sleep(3)
            stop_car()
        elif target == 17:
            AtMarker17()
            time.sleep(1.5)
            px.forward(update_speed(speed))
            time.sleep(3)
            stop_car()
        close_counter = 0
        return "close"
# MAIN LOOP
def main(headless=False):
    global active_target, close_counter, lost_counter, last_capture_time
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #Timed photo capture
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                stop_car()  # Ensure the car is stopped before capturing
                speak("Capturing image, please wait.")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"image_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                last_capture_time = current_time
                time.sleep(1)  # Brief pause for speech to complete
                
            corners, ids, _ = detector.detectMarkers(frame)
            # NO MARKER CASE
            if ids is None or len(ids) == 0:
                lost_counter += 1
                if lost_counter >= lost_threshold_frames:
                    stop_car()
                    active_target = None
                    close_counter = 0
                continue
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )
            marker_map = {int(id_val): i for i, id_val in enumerate(ids.flatten())}

            # TARGET LATCH (FIXED)
            if active_target is None:
                priority_targets = [2, 10, 12, 1, 4, 6, 11, 15, 17]
                for t in priority_targets:
                    if t in marker_map:
                        active_target = t
                        break

            # TARGET LOST
            if active_target not in marker_map:
                lost_counter += 1
                if lost_counter >= lost_threshold_frames:
                    active_target = None
                    close_counter = 0
                continue
            i = marker_map[active_target]
            lost_counter = 0
            result = track_marker_pnp(
                rvecs[i],
                tvecs[i],
                reverse_mode
            )
            if result == "close":
                active_target = None
                close_counter = 0
                stop_car()
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