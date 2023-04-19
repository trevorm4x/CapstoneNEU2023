from djitellopy import Tello
import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)

# Color of text
color = (0, 0, 255)  # red
lower_color = np.array([135, 174, 100])
upper_color = np.array([175, 222, 255])

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


def S(n, ntarget, nrange, nmax):
    dS = n - ntarget
    return (dS - nrange) / (nmax - nrange - ntarget)


def Q(S, func="linear"):
    if func == "linear":
        return S
    if func == "quadratic":
        return S * S
    if func == "sqrt":
        return S ** .5
    return 0


def V_from_Q(q, vdefault, vstepmin, vstepmax):
    return q * (vstepmax - vstepmin) + vdefault + vstepmin


def correction(
    n,
    ntarget,
    nrange,
    nmax,
    vdefault,
    vstepmin,
    vstepmax,
    func="linear"
):
    s = S(n, ntarget, nrange, nmax)
    if s < 0 or s > 1:
        return vdefault
    q = Q(s, func)
    return int(V_from_Q(q, vdefault, vstepmin, vstepmax))


if __name__ == '__main__':
    drone = Tello("10.0.0.5")

    drone.connect()

    drone.takeoff()

    # drone.change_vs_udp(22222)

    drone.streamon()

    num_frames = 3
    calibration_factor = 600
    fixed_width_pixels = None
    max_frames = 10

    frame = drone.get_frame_read().frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    frame = np.asarray(frame, dtype=np.uint8)
    cv2.imshow("Frame", frame)

    # motion parameters
    d_min = 1                      # smallest distance adjustment speed
    d_max = 30                     # largest distance adjustment speed
    x_min = 1                      # smallest sideways adjustment speed
    x_max = 25                     # largest sideways adjustment speed
    y_min = 5                      # smallest height adjustment speed
    y_max = 15                     # largest height adjustment speed
    distance_t = 10000             # optimal distance value
    distance_r = 2500              # acceptable range
    cx_t = 0.5                     # optimal x value
    cx_r = 0.0001                  # optimal x range
    cy_t = 0.5                     # optimal height value
    cy_r = 0.05                    # optimal height range
    forward_speed = 50
    time_elapsed = 0
    test_time = 5

    printouts = []
    while True:
        movescale = [0, 0, 0, 0]
        drone.send_rc_control(*movescale)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
    while True:
        if key == ord("q"):
            movescale = [0, 0, 0, 0]
            drone.send_rc_control(*movescale)
            break
        distance, cx, cy = get_distance_angle(get_frames(drone, num_frames))
        # print(distance, cx, cy)
        movescale = [0, forward_speed, 0, 0]

        if distance < 0:
            drone.send_rc_control(*movescale)
            continue

        """
        Basic formula is:
        minspeed + (maxspeed - minspeed) * (1 - relative displacment)
        where relative displacement is the percentage difference from
        current and ideal range, such that a small difference gives
        minspeed and a very large distance gives maxspeed
        """

        # s_plus = (distance - distance_t[0]) / (d_max - d_min)
        # s_minus = (distance + distance_t[1]) / (d_max - d_min)

        move_right = correction(
            cx,
            cx_t,
            cx_r,
            1,
            0,
            x_min,
            x_max
        )
        move_left = correction(
            cx,
            cx_t,
            -cx_r,
            0,
            0,
            -x_min,
            -x_max
        )
        if move_right == 0:
            movescale[3] = move_left
        else:
            movescale[3] = move_right

        drone.send_rc_control(*movescale)
        time.sleep(0.1)
        time_elapsed += 0.1
        print(f"{time_elapsed}, {distance}")
        printouts.append([time_elapsed, distance * 50])
        if time_elapsed >= test_time:
            break
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.land()
    drone.end()

    import pandas as pd
    df = pd.DataFrame(printouts, columns=["time", "distance"])
    df.to_csv("./pouts.csv")
