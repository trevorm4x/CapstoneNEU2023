from djitellopy import Tello
import cv2
import numpy as np
import time
from threading import Thread

camera = cv2.VideoCapture(0)

# Color of text
color = (0, 0, 255)  # red
lower_color = np.array([135, 174, 100])
upper_color = np.array([172, 222, 255])

# Define the dimensions of the object in cm
width = 28
height = 24


def get_distance_angle(frames):
    centroids = {}
    global fixed_width_pixels
    distances = []
    cxs = []
    cys = []
    for frame in frames:

        frame = np.asarray(frame)
        cv2.imshow("f1", frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Calculate the centroid of the object
            cx = x + w // 2
            cy = y + h // 2
            centroids[time.time()] = (cx, cy)
            centroids = {
                t: centroid for t, centroid in centroids.items()
                if t > time.time() - max_frames / 30}
            cxs.append(cx / frame.shape[0])
            cys.append(cy / frame.shape[1])

            # Estimate the distance of the object from the camera
            if fixed_width_pixels is None:
                fixed_width_pixels = w
            distance = width * calibration_factor / (w / fixed_width_pixels)
            distances.append(distance)

    if len(distances):
        return (
            sorted(distances)[int(len(distances) / 2)],
            sorted(cxs)[int(len(cxs) / 2)],
            sorted(cys)[int(len(cys) / 2)],
        )
    return -1, -1, -1


def get_frames(drone, n_frames=10):
    global frame
    frames = []
    for _ in range(n_frames):
        frame = drone.get_frame_read().frame
        frames.append(frame)
        time.sleep(1/30)
    return frames


def movedrone(
    droneIP,
    rotate_angle,
    drone_speed,
    flight_time,
    rotate_speed
):
    drone = Tello(droneIP)
    drone.connect()
    drone.takeoff()
    drone.rotate_clockwise(rotate_angle)
    time.sleep(1)
    drone.send_rc_control(0, drone_speed, 0, 0)
    print("sleeping..")
    time.sleep(flight_time)
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    drone.send_rc_control(0, 0, 0, rotate_speed)
    time.sleep(5.5)
    drone.land()
    drone.end()
    return None


if __name__ == '__main__':
    drone = Tello("10.0.0.5")

    drone.connect()

    drone.streamon()
    drone.takeoff()

    num_frames = 20
    calibration_factor = 1700
    fixed_width_pixels = None
    max_frames = 20

    frame = drone.get_frame_read().frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    frame = np.asarray(frame, dtype=np.uint8)
    cv2.imshow("Frame", frame)

    distance, _, _ = get_distance_angle(get_frames(drone, 10))
    print(f"{distance = }")

    cv2.destroyAllWindows()
    drone.streamoff()
    drone.move_up(20)
    drone.move_down(20)
    drone.land()

    cv_cm_const_r = 394
    dt = distance / cv_cm_const_r
    dl = dt * 3 ** 0.5
    drone_speed = 50
    flight_time_r = dl / drone_speed + 0.6
    # flight_time = 3
    print(f"{dt = }")
    print(f"{dl = }")
    print(f"{flight_time_r = }")

    cv_cm_const_l = 310
    dt = distance / cv_cm_const_l
    dl = dt * 3 ** 0.5
    drone_speed = 50
    flight_time_l = dl / drone_speed + 0.6
    # flight_time = 3
    print(f"{dt = }")
    print(f"{dl = }")
    print(f"{flight_time_l = }")

    movedrone_r = ["10.0.0.3", 21, drone_speed, flight_time_r, -30]
    movedrone_l = ["10.0.0.4", -21, drone_speed, flight_time_l, 30]

    Thread(target=movedrone, args=movedrone_r).start()
    Thread(target=movedrone, args=movedrone_l).start()

    drone.end()
