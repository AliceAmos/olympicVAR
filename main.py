import cv2
import time
from scipy.io import loadmat
import pandas as pd
from videos.PoseModule import PoseDetector
from videos.landmark import LandMark

FRAMES_PER_VID = 103

def load_data():
    annots = loadmat('videos/split_4_test_list/split_4_train_list.mat')
    arranged = [[element for element in upperElement] for upperElement in annots['consolidated_train_list']]
    new_data = list()
    for i in arranged:
        new_data.append((i[0], i[1], i[2]))
    columns = ['class', 'video no.', 'score']
    data = pd.DataFrame(new_data, columns=columns)

    return data

def main():

    cap = cv2.VideoCapture('videos/videos/d_001.avi')
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
        print(str(i), lmList)
        if len(lmList) > 0:
            angle = detector.calcAngle(video=img,lm1= 24, lm2=26, lm3=28)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(10)
        i += 1

    print("################################################################################################")
    for j, landmark in enumerate(inter):
        if not landmark.list and j+1 < len(inter):
            prev_lm = inter[j-1]
            next_lm = inter[j+1]
            for el1, el2 in zip(prev_lm.list, next_lm.list):
                mean = []
                landmark.list.append(mean)
                for var1, var2 in zip(el1, el2):
                    mean.append((var1+var2)//2)

    for k in range(len(inter)):

        print(str(k), inter[k].list)
    print(load_data())




if __name__ == "__main__":
    main()