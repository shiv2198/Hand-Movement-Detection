import numpy as np
import cv2
import argparse
from collections import deque



cap = cv2.VideoCapture(0)

pts = deque(maxlen=1028)

Lower_green = np.array([0, 30, 60])
'''50, 50, 110'''
Upper_green = np.array([20, 150, 255])
'''255, 255, 130'''
while True:
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, Lower_green, Upper_green)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=mask)
    cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            #cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), 10)
            #cv2.circle(img, (int(x), int(y)),int(radius) , (0, 255, 255), 10)
            cv2.circle(img, center, 5, (0, 0, 0), 5)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thick =    5 #int(np.sqrt(len(pts) / float(i + 1)) )
        cv2.line(img, pts[i - 1], pts[i], (0, 0, 225), thick)


    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    cv2.imshow("Frame", img)

    k = cv2.waitKey(30) & 0xFF
    if k == 32:
        break
# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()