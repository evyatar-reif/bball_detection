import cv2
import mediapipe as mp
import time
import sys
import PoseModule as pd
import BallModule as bd


def main():
    args = sys.argv[1:]
    count = 0
    cap = cv2.VideoCapture("videos/prof1.mp4")
    pTime = -1
    poseDetector = pd.poseDetector()
    ballDetector = bd.ballDetector()

    while True:
        (
            success,
            img,
        ) = cap.read()
        if not success:
            break

        img, landmark_list = poseDetector.findPose(img)
        img = ballDetector.findBall(img)
        print(landmark_list)
        # Display fps
        cTime = time.time()
        fps = 1 / (cTime - pTime + 0.01)
        pTime = time.time()

        # Scale
        scale_percent = 100
        if args != [] and args[0] == "-scale" and args[1] != None:
            scale_percent = int(args[1])  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.putText(
            resized,
            str(int(fps)),
            (70, 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3,
        )

        cv2.imshow("Detection", resized)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
