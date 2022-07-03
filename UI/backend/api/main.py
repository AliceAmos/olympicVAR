import pickle
import pandas as pd
from predictVid import predict

FRAMES_PER_VID = 103

# def interpolated_lankmark(prev, next, difference):
#     toReturn = []

#     for el1, el2 in zip(prev, next):
#         interp = []
#         toReturn.append(interp)
#         for idx, (var1, var2) in enumerate(zip(el1, el2)):
#             if idx == 0:
#                 interp.append(var1)
#             else:
#                 part = abs(var1 - var2) // (difference+1)
#                 if var1 > var2:
#                     interp.append(var1 - part)
#                 else:
#                     interp.append(var1 + part)
#     return toReturn

# def drawTrainersSkeleton(vidPath):
#     cap = cv2.VideoCapture(vidPath)
#     pTime = 0
#     detector = PoseDetector()
#     i = 0
#     inter = []
#     while i < FRAMES_PER_VID:
#         success, img = cap.read()
#         img = detector.findPose(img)
#         lmList = detector.getPosition(img)
#         lm_wrapper = LandMark(lmList)
#         inter.append(lm_wrapper)
#         # print(str(i), lmList)
#         if len(lmList) > 0:
#             angle = detector.calcAngle(video=img, lm1=24, lm2=26, lm3=28)

#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
#         cv2.imshow("Image", img)
#         cv2.waitKey(10)
#         i += 1

#     print("############################################## AFTER INTERPOLATION ##################################################")
#     for j, landmark in enumerate(inter):
#         if not landmark.list and j + 1 < len(inter):
#             prev_lm = inter[j - 1]
#             next_lm = inter[j + 1]
#             k = j+1
#             while not next_lm.list and k < len(inter):
#                 next_lm = inter[k]
#                 k += 1

#             diff = k - j
#             if diff > 1:
#                 diff -= 1
#             counter = 0
#             while counter < diff and j < len(inter):
#                 landmark.list = interpolated_lankmark(prev_lm.list, next_lm.list, diff-counter)
#                 counter += 1
#                 prev_lm = landmark
#                 j += 1
#                 if j < len(inter):
#                     landmark = inter[j]

#     # printing interpolated landmarks
#     for k in range(len(inter)):
#         print(str(k), inter[k].list)

#     return inter


def main():

    # dict = preprocessToClasses(load_data())
    # saveFrames(dict)

    # drawTrainersSkeleton("/Users/I555250/PycharmProjects/olympicVAR/diving/009.avi")
    #trainParts()
    # print(SomersaultsCounter.calc_somersaults(drawTrainersSkeleton("/Users/I555250/PycharmProjects/olympicVAR/diving/093.avi")))
    #print(load_data())
    #pred_vids()

    # df = pd.read_csv ('/Users/I555250/PycharmProjects/olympicVAR/new_data.csv')
    # scores = load_data()['score'].tolist()
    # df['score'] = scores
    # df.to_csv('/Users/I555250/PycharmProjects/olympicVAR/new_data.csv')

    # pred_splited_vids()
    # path = '/Users/I555250/PycharmProjects/olympicVAR/new_data.csv'
    # trainMLmodels(path)




    loaded_model_path = 'model/random_forest_model.sav'
    loaded_model = pickle.load(open(loaded_model_path, 'rb'))

    frames_predicton_model_path = 'model/extended.model'
    new_vid_scores = predict('https://res.cloudinary.com/dtedceoh4/video/upload/v1654449663/videos/015_ahosnt.avi', frames_predicton_model_path)
    new_vid_scores = new_vid_scores[:103]
    new_vid_scores.insert(0, 9)
    new_vid_scores.insert(1, 9)
    vid_df = pd.DataFrame(data=[new_vid_scores])
    print(vid_df)
    y_new_vid = loaded_model.predict(vid_df)
    print("Predicted score: ", y_new_vid)

if __name__ == "__main__":
    main()