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
    drone1 = Tello("10.0.0.3")
    drone1.connect()
    drone1.takeoff()
    drone1.change_vs_udp(22222)
    drone1.streamon()

    drone2 = Tello("10.0.0.5")
    drone2.connect()
    drone2.takeoff()
    drone2.change_vs_udp(33333)
    drone2.streamon()

    num_frames = 10
    calibration_factor = 600
    fixed_width_pixels = None
    max_frames = 10

    frame1 = drone1.get_frame_read().frame
    frame2 = drone2.get_frame_read().frame

    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_color, upper_color)

    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_color, upper_color)

    frame1 = np.asarray(frame1, dtype=np.uint8)
    frame2 = np.asarray(frame2, dtype=np.uint8)
    cv2.imshow("Frame", frame1)
    cv2.imshow("Frame", frame2)

    # motion parameters
    d1_rules = dict(
        d_min=5,                      # smallest distance adjustment speed
        d_max=30,                     # largest distance adjustment speed
        x_min=5,                      # smallest sideways adjustment speed
        x_max=15,                     # largest sideways adjustment speed
        y_min=5,                      # smallest height adjustment speed
        y_max=15,                     # largest height adjustment speed
        distance_t=7500,             # optimal distance value
        distance_r=500,              # acceptable range
        cx_t=0.7,                     # optimal x value
        cx_r=0.05,                    # optimal x range
        cy_t=0.7,                     # optimal height value
        cy_r=0.05                    # optimal height range
    )

    # motion parameters
    d2_rules = dict(
        d_min=5,                      # smallest distance adjustment speed
        d_max=30,                     # largest distance adjustment speed
        x_min=5,                      # smallest sideways adjustment speed
        x_max=15,                     # largest sideways adjustment speed
        y_min=5,                      # smallest height adjustment speed
        y_max=15,                     # largest height adjustment speed
        distance_t=12500,             # optimal distance value
        distance_r=500,              # acceptable range
        cx_t=0.3,                     # optimal x value
        cx_r=0.05,                    # optimal x range
        cy_t=0.3,                     # optimal height value
        cy_r=0.05                    # optimal height range
    )

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            movescale = [0, 0, 0, 0]
            drone1.send_rc_control(*movescale)
            drone2.send_rc_control(*movescale)
            break
        distance1, cx1, cy1 = get_distance_angle(get_frames(drone1, 10))
        distance2, cx2, cy2 = get_distance_angle(get_frames(drone2, 10))
        print(distance1, cx1, cy1)
        print(distance1, cx1, cy1)
        movescale1 = [0, 0, 0, 0]
        movescale2 = [0, 0, 0, 0]

        if distance1 < 0:
            drone1.send_rc_control(*movescale1)
            continue
        if distance2 < 0:
            drone2.send_rc_control(*movescale2)
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
            d1_rules["cx"],
            d1_rules["cx_t"],
            d1_rules["cx_r"],
            1,
            0,
            d1_rules["x_min"],
            d1_rules["x_max"]
        )
        move_left = correction(
            d1_rules["cx"],
            d1_rules["cx_t"],
            -d1_rules["cx_r"],
            0,
            0,
            -d1_rules["x_min"],
            -d1_rules["x_max"]
        )
        if move_right == 0:
            movescale1[0] = move_left
        else:
            movescale1[0] = move_right

        move_up = correction(
            d1_rules["cy"],
            d1_rules["cy_t"],
            -d1_rules["cy_r"],
            0,
            0,
            d1_rules["y_min"],
            d1_rules["y_max"]
        )
        move_down = correction(
            d1_rules["cy"],
            d1_rules["cy_t"],
            d1_rules["cy_r"],
            1,
            0,
            -d1_rules["y_min"],
            -d1_rules["y_max"]
        )
        if move_up == 0:
            movescale1[2] = move_down
        else:
            movescale1[2] = move_up

        move_forward = correction(
            d1_rules["distance"],
            d1_rules["distance_t"],
            d1_rules["distance_r"],
            100_000,
            0,
            d1_rules["d_min"],
            d1_rules["d_max"]
        )
        move_back = correction(
            d1_rules["distance"],
            d1_rules["distance_t"],
            -d1_rules["distance_r"],
            0,
            0,
            -d1_rules["d_min"],
            -d1_rules["d_max"]
        )
        if move_forward == 0:
            movescale1[1] = move_back
        else:
            movescale1[1] = move_forward

        move_right = correction(
            d2_rules["cx"],
            d2_rules["cx_t"],
            d2_rules["cx_r"],
            1,
            0,
            d2_rules["x_min"],
            d2_rules["x_max"]
        )
        move_left = correction(
            d2_rules["cx"],
            d2_rules["cx_t"],
            -d2_rules["cx_r"],
            0,
            0,
            -d2_rules["x_min"],
            -d2_rules["x_max"]
        )
        if move_right == 0:
            movescale2[0] = move_left
        else:
            movescale2[0] = move_right

        move_up = correction(
            d2_rules["cy"],
            d2_rules["cy_t"],
            -d2_rules["cy_r"],
            0,
            0,
            d2_rules["y_min"],
            d2_rules["y_max"]
        )
        move_down = correction(
            d2_rules["cy"],
            d2_rules["cy_t"],
            d2_rules["cy_r"],
            1,
            0,
            -d2_rules["y_min"],
            -d2_rules["y_max"]
        )
        if move_up == 0:
            movescale2[2] = move_down
        else:
            movescale2[2] = move_up

        move_forward = correction(
            d2_rules["distance"],
            d2_rules["distance_t"],
            d2_rules["distance_r"],
            100_000,
            0,
            d2_rules["d_min"],
            d2_rules["d_max"]
        )
        move_back = correction(
            d2_rules["distance"],
            d2_rules["distance_t"],
            -d2_rules["distance_r"],
            0,
            0,
            -d2_rules["d_min"],
            -d2_rules["d_max"]
        )
        if move_forward == 0:
            movescale2[1] = move_back
        else:
            movescale2[1] = move_forward

        drone1.send_rc_control(*movescale1)
        drone2.send_rc_control(*movescale2)

    cv2.destroyAllWindows()
    drone1.streamoff()
    drone2.streamoff()
    drone1.land()
    drone2.land()
    drone1.end()
    drone2.end()
