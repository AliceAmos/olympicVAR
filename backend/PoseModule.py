from math import degrees, atan2
import cv2
import mediapipe as mp

class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=self.upBody,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
    def findPose(self, img, draw=True):                                                                                 # draw landmarks on the image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):                                                                              # get the landmarks of the person in the image and return a list of them
        self.lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList

    def calcAngle(self, video, lm1, lm2, lm3, draw=True):                                                               # calculate the angle between 3 passed body-parts numbers and draw their points

        x1, y1 = self.lmList[lm1][1:]
        x2, y2 = self.lmList[lm2][1:]
        x3, y3 = self.lmList[lm3][1:]

        if draw:
            cv2.circle(video, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(video, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(video, (x3, y3), 5, (255, 0, 255), cv2.FILLED)

        deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360                                                           # calculate the degree
        deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
        return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)





