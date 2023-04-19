import cv2
from djitellopy import Tello
import argparse
import imutils
from collections import deque
import numpy as np
import time
import pandas as pd
from threading import Thread


def display_video(capture):
    while (True):
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        print("waiting key")
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


lower_1 = np.asarray([0, 100, 100])
upper_1 = np.asarray([50, 255, 255])

on = True
cap = cv2.VideoCapture(0)
while (on):
    ret, frame = cap.read()

    interest_area = 0.10
    frame = np.asarray(frame)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    left_bound = frame_hsv.shape[0] // 2 - int(frame_hsv.shape[0] * interest_area)
    right_bound = frame_hsv.shape[0] // 2 + int(frame_hsv.shape[0] * interest_area)
    upper_bound = frame_hsv.shape[1] // 2 - int(frame_hsv.shape[1] * interest_area)
    lower_bound = frame_hsv.shape[1] // 2 + int(frame_hsv.shape[1] * interest_area)

    mask1 = cv2.inRange(hsv, lower_1, upper_1)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)

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
    key = cv2.waitKey(1) & 0xFF
    if key == ord("f"):
        interest_area = frame_hsv[
                        frame_hsv.shape[0] // 2 - int(frame_hsv.shape[0] * interest_area):
                        frame_hsv.shape[0] // 2 + int(frame_hsv.shape[0] * interest_area),
                        frame_hsv.shape[1] // 2 - int(frame_hsv.shape[1] * interest_area):
                        frame_hsv.shape[1] // 2 + int(frame_hsv.shape[1] * interest_area)
                        ]

        # color_value = interest_area[:,:,0].sum()
        color_values = interest_area.reshape(-1, 3).tolist()

        interest_dataframe = pd.DataFrame(color_values).describe().T
        interest_dataframe = interest_dataframe.drop(columns=['count'])
        mean = interest_dataframe["mean"].values//1
        std = interest_dataframe["std"].values//1

        lower_1[0] = mean[0]-std[0]

        lower_1[1] = mean[1] - 2*std[1]

        upper_1[0] = mean[0] + std[0]

        upper_1[1] = mean[1] + 2*std[1]


        print(f'{interest_dataframe["mean"]}')
        print(f'{lower_1=}')
        print(f'{upper_1=}')

    cv2.rectangle(frame, (upper_bound, left_bound), (lower_bound, right_bound), (0,255,0))
    frame = cv2.flip(frame, 1)
    mask1 = cv2.flip(mask1, 1)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
