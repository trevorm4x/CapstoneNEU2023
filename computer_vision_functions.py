import cv2
import numpy as np
from collections import deque
import argparse
import time
import imutils


def get_object_pos_and_dist(eval_frames=5, cap=0, target_color="RED", radius_size=10):
    if cap == 0:
        return -1
    else:
        # Check color to view, and choose HSV
        if target_color.lower() == 'red' or target_color.lower() == 'r':
            lower = (0, 150, 20)
            upper = (10, 255, 255)
        elif target_color.lower() == 'blue' or target_color.lower() == 'b':
            lower = (100, 0, 20)
            upper = (120, 255, 255)
        else:
            print(f'ERROR: INVALID TARGET COLOR\n')
            return -1

        frame_count = 0

        radius_count = 0
        radius_record = 0

        center_count = 0
        center_record = np.array([0, 0])

        pts = deque(maxlen=64)

        while frame_count < eval_frames:
            _, frame = cap.read()
            frame_count += 1

            # Image processing techniques to reduce noise
            frame = imutils.resize(frame, width=600)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Grab pixels in color range
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get contours
            cnts = imutils.grab_contours(cnts)

            if len(cnts) > 0:
                # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # only proceed if the radius meets a minimum size
                if radius > radius_size:
                    # Record measured radius
                    radius_record += radius
                    radius_count += 1

                    # Record measured center
                    center_record[0] += center[0]
                    center_record[1] += center[1]
                    center_count += 1

                    # draw the circle and centroid on the frame0,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the points queue
            pts.appendleft(center)
            # loop over the set of tracked points

            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        center_return = np.array([center_record[0] / center_count, center_record[1] / center_count])
        radius_return = radius_record / radius_count
        return center_return, radius_return
