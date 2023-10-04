import cv2
import mediapipe as mp


class poseDetector:
    def __init__(
        self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5
    ):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            # self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon
        )
        self.landmarks = []

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
            pass

        # draw circles and connectors
        self.mpDraw.draw_landmarks(
            img,
            results.pose_landmarks,
            self.mpPose.POSE_CONNECTIONS,
            self.mpDraw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=5),
            self.mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=5),
        )

        return img
