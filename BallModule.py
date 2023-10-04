import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class ballDetector:
    def __init__(self):
        self.prevCircle = None
        self.dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

    def findBall(self, img, draw=True):
        grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)

        circles = cv2.HoughCircles(
            blurFrame,
            cv2.HOUGH_GRADIENT,
            1.2,
            100,
            param1=100,
            param2=30,
            minRadius=25,
            maxRadius=75,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None

            for i in circles[0, :]:
                if chosen is None:
                    chosen = i
                if self.prevCircle is not None:
                    if self.dist(
                        chosen[0], chosen[1], self.prevCircle[0], self.prevCircle[1]
                    ) <= self.dist(i[0], i[1], self.prevCircle[0], self.prevCircle[1]):
                        chosen = i

            cv2.circle(img, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
            cv2.circle(img, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
            self.prevCircle = chosen

        return img
