from PoseModule import *
from moviepy.editor import *
import cv2
import time
from landmark import LandMark


FRAMES_PER_VID = 103

def cutVid():

    delimiter = 0.25
    clips = []
    timer = 0
    clip = VideoFileClip('/Users/I555250/PycharmProjects/olympicVAR/diving/002.avi')
    cap = cv2.VideoCapture('/Users/I555250/PycharmProjects/olympicVAR/diving/002.avi')
    pTime = 0
    detector = PoseDetector()
    i = 0
    inter = []
    somer = 0
    counter = 0
    inSault = False
    while i < FRAMES_PER_VID:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        if len(lmList) > 0:
            if lmList[1][1] < lmList[11][1]:
                inSault = True
                if somer < 1:
                    somer += 1
            else:
                if inSault == True:
                    somer -= 1
                    counter += 1
                inSault = False
        lm_wrapper = LandMark(lmList)
        inter.append(lm_wrapper)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        i+=1
    if somer == 1:
        counter += 0.5

    print(counter)
def showVid(path):
    cap = cv2.VideoCapture(path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()