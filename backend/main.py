import time

from PoseModule import PoseDetector
from SomersaultsCounter import calc_somersaults
from createData import *
from landmark import LandMark

FRAMES_PER_VID = 103

def interpolated_lankmark(prev, next, difference):                                                                      # interpolate in order to maintain consistency of thr landmarks
    toReturn = []

    for el1, el2 in zip(prev, next):
        interp = []
        toReturn.append(interp)
        for idx, (var1, var2) in enumerate(zip(el1, el2)):                                                              # go over 2 single landmarks (one from before empty, one after it)
            if idx == 0:                                                                                                # first element of the landmark is the body part no. (no change)
                interp.append(var1)
            else:
                part = abs(var1 - var2) // (difference+1)                                                               # complete the missing parts with linear fomula (according to the number of missing parts in between)
                if var1 > var2:
                    interp.append(var1 - part)
                else:
                    interp.append(var1 + part)
    return toReturn

def drawTrainersSkeleton(vidPath):                                                                                      # method to identify and mark the skeleton of a trainer on the passed video
    cap = cv2.VideoCapture(vidPath)
    pTime = 0
    detector = PoseDetector()                                                                                           # init the class of positions and landmarks
    i = 0
    inter = []
    while i < FRAMES_PER_VID:
        success, img = cap.read()                                                                                       # read each frame of the video
        img = detector.findPose(img)
        lmList = detector.getPosition(img)                                                                              # get landmarks' list
        lm_wrapper = LandMark(lmList)
        inter.append(lm_wrapper)

        if len(lmList) > 0:
            angle = detector.calcAngle(video=img, lm1=24, lm2=26, lm3=28)                                               # calculate the angle between the inner leg (indicates of jump and sommersault)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(10)
        i += 1

    print("############################################## EXECUTING INTERPOLATION ##################################################")
    for j, landmark in enumerate(inter):                                                                                # go over landmarks' list
        if not landmark.list and j + 1 < len(inter):                                                                    # empty list means no position was recognized - need interpolation
            prev_lm = inter[j - 1]
            next_lm = inter[j + 1]
            k = j+1
            while not next_lm.list and k < len(inter):                                                                  # check the amount of empty lists in between
                next_lm = inter[k]
                k += 1

            diff = k - j
            if diff > 1:
                diff -= 1
            counter = 0
            while counter < diff and j < len(inter):
                landmark.list = interpolated_lankmark(prev_lm.list, next_lm.list, diff-counter)                         # interpolate
                counter += 1
                prev_lm = landmark
                j += 1
                if j < len(inter):
                    landmark = inter[j]

                                                                                                                        # printing interpolated landmarks
    # for k in range(len(inter)):
    #     print(str(k), inter[k].list)

    return inter


def main():

    loaded_model_path = 'model/random_forest_model.sav'
    loaded_model = pickle.load(open(loaded_model_path, 'rb'))

    frames_predicton_model_path = 'model/extended.model'
    new_vid_scores = predict('new_vid_91.mp4', frames_predicton_model_path)
    new_vid_scores = new_vid_scores[:103]
    new_vid_scores.append(calc_somersaults(drawTrainersSkeleton('new_vid_91.mp4')))
    vid_df = pd.DataFrame(data=[new_vid_scores])
    y_new_vid = loaded_model.predict(vid_df)
    print("Predicted score: ", y_new_vid)

if __name__ == "__main__":
    main()