from djitellopy import Tello
import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)

# Color of text
color = (0, 0, 255)  # red
lower_color = np.array([155, 174, 100])
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
        cv2.imshow("Frame", frame)
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


if __name__ == '__main__':

    drone = Tello('10.0.0.2')

    drone.connect()

    drone.takeoff()

    drone.change_vs_udp(22222)

    drone.streamon()

    num_frames = 10
    calibration_factor = 600
    fixed_width_pixels = None
    max_frames = 10

    frame = drone.get_frame_read().frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    frame = np.asarray(frame, dtype=np.uint8)
    cv2.imshow("Frame", frame)

    distance = 100000

    while 1:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        distance, cx, cy = get_distance_angle(get_frames(drone, 10))
        print(distance, cx, cy)
        if distance < 1:
            ...
        elif distance > 10_000:
            drone.move_forward(20)
        elif distance < 6_000:
            drone.move_back(20)
        elif cx > .7:
            drone.move_right(20)
        elif cx < .3:
            drone.move_left(20)
        else:
            ...
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.land()
    drone.end()
