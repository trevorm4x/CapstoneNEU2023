from djitellopy import Tello
import cv2
import numpy as np
import time
from collections import deque

# Color of text
color = (0, 0, 255)  # red
'''
lower_color = np.array([40, 20, 20])
upper_color = np.array([80, 255, 255])
'''
lower_color = np.array([150, 150, 100])
upper_color = np.array([175, 225, 255])
# Define the dimensions of the object in cm
width = 28


# height = 24


def get_distance_angle(frames):
    centroids = {}
    global fixed_width_pixels
    distances = []
    cxs = []
    cys = []
    for frame in frames:

        frame = np.asarray(frame)
        cv2.imshow("FrameE9", frame)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
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
            cys.append(cy / frame.shape[0])
            cxs.append(cx / frame.shape[1])

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
        time.sleep(1 / 30)
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
    drone = Tello()

    drone.connect()
    #drone.change_vs_udp(11111)
    drone.streamon()
    drone.takeoff()
    landed = False

    num_frames = 2
    calibration_factor = 600
    fixed_width_pixels = None
    max_frames = 2

    frame = drone.get_frame_read().frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    frame = np.asarray(frame, dtype=np.uint8)
    cv2.imshow("FrameE91", frame)

    # motion parameters
    d_min = 2  # smallest distance adjustment speed
    d_max = 10  # largest distance adjustment speed
    x_min = 5  # smallest sideways adjustment speed
    x_max = 20  # largest sideways adjustment speed
    yaw_min = 10
    yaw_max = 20
    y_min = 5  # smallest height adjustment speed
    y_max = 20  # largest height adjustment speed
    distance_t = 7000  # optimal distance value
    distance_r = 500  # acceptable range
    distance_max = 20000000
    cx_t_move = 0.5  # optimal x value
    cx_t_move_range = 0.02
    cx_t_rotate = 1
    cy_t = 0.5 # optimal height value
    cy_t_range = 0.01
    direction = 1
    distance_window_size = 5
    number_valid_distances = 1
    land = False

    danger_zone = 0.05
    recent_list = [0] * distance_window_size
    recent_list = deque(recent_list)

    tag_found = True

    target_degree = 0
    distance_to_tag = 10000
    actual_speed = 73.83
    drone.rotate_counter_clockwise(target_degree)
    time_to_fly = distance_to_tag/actual_speed
    time_elapsed = 0
    time_per_loop = 0.5
    drone.send_rc_control(0, 50, 0, 0)

    while time_elapsed < time_to_fly:

        distance, cx, cy = get_distance_angle(get_frames(drone, num_frames))
        if not distance == -1:
            break
        time_elapsed += time_per_loop
        time.sleep(time_per_loop)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            movescale = [0, 0, 0, 0]
            drone.streamoff()
            drone.send_rc_control(*movescale)
            break
        distance, cx, cy = get_distance_angle(get_frames(drone, num_frames))

        if distance > 0 and number_valid_distances < 2 * distance_window_size and landed == False:
            number_valid_distances += 1
        elif number_valid_distances == 2 * distance_window_size and landed == False:
            number_valid_distances = 2 * distance_window_size
        else:
            number_valid_distances = 1

        recent_list.pop()
        recent_list.appendleft(distance)
        print(recent_list)
        print(distance, cx, cy)
        movescale = [0, 0, 0, 0]
        if sum(recent_list) == -distance_window_size and landed == False:
            print(f'SPINNING')
            movescale = [0, 0, 0, 0]
            movescale[3] = direction * yaw_max
            drone.send_rc_control(*movescale)
            continue

        if landed == False and number_valid_distances >= distance_window_size and land == True:
            if distance_t - distance_r <= sum(recent_list) / distance_window_size and sum(
                    recent_list) / distance_window_size <= distance_t + distance_r:
                landed = True
                drone.land()

        if landed == True and sum(recent_list) == -distance_window_size and land == True:
            drone.takeoff()
            landed = False

        if cy < danger_zone or cy > (1 - danger_zone) or distance > distance_max:
            drone.send_rc_control(*movescale)
            continue
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
            cx_t_move,
            cx_t_move_range,
            1,
            0,
            x_min,
            x_max
        )
        move_left = correction(
            cx,
            cx_t_move,
            -cx_t_move_range,
            0,
            0,
            -x_min,
            -x_max
        )
        if move_right == 0:
            movescale[0] = move_left
        else:
            movescale[0] = move_right

        move_up = correction(
            cy,
            cy_t,
            -cy_t_range,
            0,
            0,
            y_min,
            y_max
        )
        move_down = correction(
            cy,
            cy_t,
            cy_t_range,
            1,
            0,
            -y_min,
            -y_max
        )
        if move_up == 0:
            movescale[2] = move_down
        else:
            movescale[2] = move_up

        move_forward = correction(
            distance,
            distance_t,
            distance_r,
            100_000,
            0,
            d_min,
            d_max
        )
        move_back = correction(
            distance,
            distance_t,
            -distance_r,
            0,
            0,
            -d_min,
            -d_max
        )
        if move_forward == 0:
            movescale[1] = move_back
        else:
            movescale[1] = move_forward

        """
        if distance < distance_t[0]:
            movescale[1] = int(d_min - (d_max - d_min) * (
                1 - distance / distance_t[0]
            ))

        if distance < distance_t - distance_r:
            movescale[1] = int(d_min + (d_max - d_min) * (
                    1 - distance_t / distance
            ))

        if cx < (cx_t_move - cx_t_move_range) and cx > (1 - cx_t_rotate):
            movescale[0] = -int(x_min + (x_max - x_min) * (
                    1 - cx / cx_t_move
            ))
        if cx > (cx_t_move + cx_t_move_range) and cx < cx_t_rotate:
            movescale[0] = int(x_min + (x_max - x_min) * (
                    1 - cx_t_move / cx
            ))

        if cx < (1 - cx_t_rotate):
            direction = -1
            movescale[3] = -int(yaw_min + (yaw_max - yaw_min) * (
                    1 - cx / cx_t_move
            ))
        if cx > cx_t_rotate:
            direction = 1
            movescale[3] = int(yaw_min + (yaw_max - yaw_min) * (
                    1 - cx_t_move / cx
            ))

        if cy < cy_t - cy_t_range:
            movescale[2] = int(y_min + (y_max - y_min) * (
                    1 - cy / cy_t
            ))

        if cy > cy_t + cy_t_range:
            movescale[2] = -int(y_min + (y_max - y_min) * (
                    1 - cy_t / cy
            ))

        drone.send_rc_control(*movescale)
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.land()
    drone.end()
    """
