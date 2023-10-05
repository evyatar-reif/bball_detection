import cv2
import mediapipe as mp
import numpy as np


class poseDetector:
    def __init__(
        self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5
    ):
        # self.mode = mode
        # self.upBody = upBody
        # self.smooth = smooth
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            # self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon
        )
        self.landmarks = []
        self.bend_arr = []

    def findPose(self, img, draw=True):
        # color it in rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        # look for landmarks
        results = self.pose.process(img)

        # recolor to bgr
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        try:
            self.landmarks = results.pose_landmarks.landmark
        except:
            return img

        # draw circles and connectors
        self.mpDraw.draw_landmarks(
            img,
            results.pose_landmarks,
            self.mpPose.POSE_CONNECTIONS,
            self.mpDraw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=5),
            self.mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=5),
        )

        self.set_landmarks()
        self.display_angle(img, self.left_elbow, self.left_wrist, self.left_index)
        self.display_angle(img, self.right_elbow, self.right_wrist, self.right_index)

        return img

    def set_landmarks(self):
        self.left_shoulder = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        self.right_shoulder = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        self.left_elbow = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        self.right_elbow = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value].y,
        ]
        self.left_wrist = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].y,
        ]
        self.right_wrist = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value].y,
        ]
        self.left_pinky = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_PINKY.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_PINKY.value].y,
        ]
        self.right_pinky = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_PINKY.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_PINKY.value].y,
        ]
        self.left_index = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_INDEX.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_INDEX.value].y,
        ]
        self.right_index = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_INDEX.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_INDEX.value].y,
        ]
        self.left_thumb = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_THUMB.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_THUMB.value].y,
        ]
        self.right_thumb = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_THUMB.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_THUMB.value].y,
        ]
        self.left_hip = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].y,
        ]
        self.right_hip = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        self.left_knee = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_KNEE.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        self.right_knee = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_KNEE.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        self.left_ankle = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_ANKLE.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        self.right_ankle = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_ANKLE.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]
        self.left_heel = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_HEEL.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_HEEL.value].y,
        ]
        self.right_heel = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_HEEL.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_HEEL.value].y,
        ]
        self.left_foot_index = [
            self.landmarks[self.mpPose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
            self.landmarks[self.mpPose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
        ]
        self.right_foot_index = [
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
            self.landmarks[self.mpPose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
        ]

    def display_angle(self, img, a, b, c):
        angle = self.calculate_angle(a, b, c)
        height, width, channel = img.shape
        cv2.putText(
            img,
            str(angle),
            (int(width * b[0]), int(height * b[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle
