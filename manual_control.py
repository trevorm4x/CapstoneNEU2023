from djitellopy import Tello
import cv2
import numpy as np
import time

color = (0, 0, 255)
if __name__ == '__main__':

    lower_color = np.array([130, 120, 100])  # Green
    upper_color = np.array([165, 255, 255])

    d_min = 30  # smallest distance adjustment speed
    d_max = 15  # largest distance adjustment speed
    x_min = 25 # smallest sideways adjustment speed
    x_max = 15  # largest sideways adjustment speed
    yaw_min = 30
    yaw_max = 20
    y_min = 30  # smallest height adjustment speed
    y_max = 20  # largest height adjustment speed
    distance_t = 7000  # optimal distance value
    distance_r = 500  # acceptable range
    distance_max = 20000000
    cx_t_move = 0.5  # optimal x value
    cx_t_move_range = 0.005
    cx_t_rotate = 0.85
    cy_t = 0.4 # optimal height value
    cy_t_range = 0.001
    direction = 1
    distance_window_size = 5
    number_valid_distances = 1
    land = False

    tello = Tello()
    tello.connect()
    tello.streamon()
    tello.get_battery()
    tello.takeoff()

    frame = tello.get_frame_read().frame
    movescale = [0, 0, 0, 0]
    while True:


        frame = tello.get_frame_read().frame
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('MainDrone',frame)
        cv2.imshow('MaskView', mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("j"):
            movescale = [0, 0, 0, 0]
            tello.streamoff()
            tello.send_rc_control(*movescale)
            break
        if key == ord('w'):
            movescale[1] = d_min
        if key == ord('s'):
            movescale[1] = -d_min
        if key == ord('a'):
            movescale[0] = -x_min
        if key == ord('d'):
            movescale[0] = +x_min
        if key == ord('r'):
            movescale[2] = y_min
        if key == ord('f'):
            movescale[2] = -y_min
        if key == ord('q'):
            movescale[3] = -yaw_min
        if key == ord('e'):
            movescale[3] = yaw_min
        if key == ord('k'):
            movescale = [0, 0, 0, 0]

        if key == ord('l'):
            lower_color[1] -=1
            print(f'{lower_color=}')
        if key == ord('o'):
            lower_color[1] += 1
            print(f'{lower_color=}')

        print(f'command {movescale=}')
        tello.send_rc_control(*movescale)
    cv2.destroyAllWindows()
    tello.streamoff()
    tello.land()
    tello.end()



