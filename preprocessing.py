import cv2
from scipy.io import loadmat
import pandas as pd

def load_data():
    annots_train = loadmat('/Users/I555250/PycharmProjects/olympicVAR/split_4_test_list/split_4_train_list.mat')
    arranged_train = [[element for element in upperElement] for upperElement in annots_train['consolidated_train_list']]
    new_data_train = list()
    for i in arranged_train:
        if(i[0] == 1.0):
            new_data_train.append((i[0], i[1], i[2]))
    columns = ['class', 'video no.', 'score']
    data_train = pd.DataFrame(new_data_train, columns=columns)

    annots_test = loadmat('/Users/I555250/PycharmProjects/olympicVAR/split_4_test_list/split_4_test_list.mat')
    arranged_test = [[element for element in upperElement] for upperElement in annots_test['consolidated_test_list']]
    new_data_test = list()
    for j in arranged_test:
        if(j[1] == 1.0):
            new_data_test.append((j[0], j[1], j[2]))
    columns = ['class', 'video no.', 'score']
    data_test = pd.DataFrame(new_data_test, columns=columns)
    new_data = [data_train, data_test]
    new_data = pd.concat(new_data)
    new_data = new_data.sort_values(["class", "video no."])

    return new_data


def preprocessToClasses(dataFrame):

    vids_dict = {}

    for index, row in dataFrame.iterrows():
        vids_dict[int(row['video no.'])] = int(row['score'] // 10)

    return vids_dict

def saveFrames(videos_dict):

    for index in videos_dict:
        path = '/Users/I555250/PycharmProjects/olympicVAR/'
        if index == 1:
            continue
        if index < 10:
            path += "diving/00" + str(index) + ".avi"
        elif index < 100:
            path += "diving/0" + str(index) + ".avi"
        else:
            path += "diving/" + str(index) + ".avi"

        cap = cv2.VideoCapture(path)
        frames = []
        ret = True
        j = 0
        while ret:
            ret, img = cap.read()                                                                                       # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                frames.append(img)
                cv2.imwrite('/Users/I555250/PycharmProjects/olympicVAR/data/' + str(videos_dict[index]) + '/vid' + str(index) + "frame" + str(j) + '.jpg', img)
                j += 1

