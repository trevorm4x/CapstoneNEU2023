from computer_vision_functions import get_object_pos_and_dist
import cv2
from djitellopy import Tello
import argparse
import imutils
from collections import deque
import numpy as np
import time

lower_red = (0, 150, 50)
upper_red = (10, 255, 255)

lower_blue = (100, 0, 20)
upper_blue = (120, 255, 255)

lower3 = (2, 50, 55)
upper3 = (5, 255, 255)
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

if __name__ == '__main__':
    tello = Tello()

    tello.connect()

    tello.streamon()
    frame_read = tello.get_frame_read()

    while True:
        frame = frame_read.frame
        time.sleep(1/30)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)

        mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)

        mask3 = cv2.inRange(hsv, lower3, upper3)
        mask3 = cv2.erode(mask3, None, iterations=2)
        mask3 = cv2.dilate(mask3, None, iterations=2)

        cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            # find the largest contour in the mask1, then use it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame0,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # show the frame0 to our screen
        cv2.imshow("Frame", frame)
        cv2.imshow('Mask1', mask1)
        cv2.imshow('Mask2', mask2)
        cv2.imshow('Mask3', mask3)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()