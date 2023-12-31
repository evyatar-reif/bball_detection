import cv2
import mediapipe as mp
import time
import sys
import PoseModule as pd
import BallModule as bd
import matplotlib.pyplot as plt
import numpy as np


FILE_NAME = "videos/dribble4.mp4"


def main():
    pTime = 0
    cap = cv2.VideoCapture(FILE_NAME)
    poseDetector = pd.poseDetector()
    ballDetector = bd.ballDetector()

    while True:
        (
            success,
            img,
        ) = cap.read()
        if not success:
            break

        img = poseDetector.findPose(img)
        img = ballDetector.findBall(img)

        scaled_img = scale(img)
        img_with_fps = display_fps(scaled_img, pTime)

        cv2.imshow(FILE_NAME, img_with_fps)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def plot_graph(y, x=None):
    ypoints = y
    xpoints = [i for i in range(len(y))]
    plt.plot(xpoints, ypoints)
    plt.show()


def scale(img):
    args = sys.argv[1:]
    scale_percent = 100
    if args != [] and args[0] == "-scale" and args[1] != None:
        scale_percent = int(args[1])  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


def display_fps(img, pTime):
    cTime = time.time()

    fps = 1 / (cTime - pTime)
    pTime = time.time()
    cv2.putText(
        img,
        str(int(fps)),
        (70, 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3,
    )
    return img


if __name__ == "__main__":
    main()
