import cv2
import cv2.aruco as aruco
import numpy as np
from core.config import Config

if __name__ == "__main__":
    config = Config.get_instance()
    
    # --- ArUco Setup ---
    aruco_dict = config.aruco_dict
    aruco_params = config.aruco_params

    cam_matrix = config.cam_matrix
    dist_coeffs = config.dist_coeffs
    marker_size = config.marker_size
    
    TARGET_ID = 8  # <-- only track marker 8

    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    cap = cv2.VideoCapture(config.camera_id)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Define object points (marker corners in its own frame)
    obj_points = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            for i in range(len(ids)):
                marker_id = int(ids[i])

                if marker_id != TARGET_ID:
                    continue  # ignore all other markers

                # Solve pose
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corners[i], cam_matrix, dist_coeffs
                )

                if not success:
                    continue

                # Extract coordinates (camera frame)
                x, y, z = tvec.flatten()

                # Print to terminal
                print(f"Marker {TARGET_ID} position (camera frame): "
                      f"X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

                # Draw axes on marker
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, marker_size)

                # Display coordinates on screen
                center_x = int(np.mean(corners[i][0][:, 0]))
                center_y = int(np.mean(corners[i][0][:, 1])) - 20

                text = f"ID 8: X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
                cv2.putText(frame, text, (center_x, center_y),
                            font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Marker 8 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()