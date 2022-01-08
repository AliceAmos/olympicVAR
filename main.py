import cv2
import time
from PoseModule import PoseDetector
from landmark import LandMark
from trainingModel import train
from predictVid import predict
from preprocessing import load_data, preprocessToClasses, saveFrames

FRAMES_PER_VID = 103

def drawTrainersSkeleton(vidPath):
    cap = cv2.VideoCapture(vidPath)
    pTime = 0
    detector = PoseDetector()
    i = 0
    inter = []
    while i < FRAMES_PER_VID:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        lm_wrapper = LandMark(lmList)
        inter.append(lm_wrapper)
        # print(str(i), lmList)
        if len(lmList) > 0:
            angle = detector.calcAngle(video=img, lm1=24, lm2=26, lm3=28)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(10)
        i += 1

    print("############################################## AFTER INTERPOLATION ##################################################")
    for j, landmark in enumerate(inter):
        if not landmark.list and j + 1 < len(inter):
            prev_lm = inter[j - 1]
            next_lm = inter[j + 1]
            for el1, el2 in zip(prev_lm.list, next_lm.list):
                mean = []
                landmark.list.append(mean)
                for var1, var2 in zip(el1, el2):
                    mean.append((var1 + var2) // 2)

    # for k in range(len(inter)):
    #
    #     print(str(k), inter[k].list)


def main():

    # dict = preprocessToClasses(load_data())
    # saveFrames(dict)

    #try_test()
    #train()
    #drawTrainersSkeleton("new_vid_91.mp4")
    #train()
    predict()

if __name__ == "__main__":
    main()