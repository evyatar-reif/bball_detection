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

    def findPose(self, img, draw=True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRgb)
        landmark_list = []
        if draw and results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
            )
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                # draw circle in point coordinate
                height, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                landmark_list.append({"id": id, "landmark": landmark})

        return img, landmark_list
