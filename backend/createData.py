import SomersaultsCounter
from main import drawTrainersSkeleton
from predictVid import *
from preprocessing import *


def pred_vids():                                                                                                        # method to create a new .csv file - to be a data file for ML algorithm
                                                                                                                        # predicting according to the trained model each video from our dataset
    all_vids_preds = list()
    for index in range(1, 371):
        path = ''                                                                                                       # put your root dir
        if index < 10:
            path += "diving/00" + str(index) + ".avi"
        elif index < 100:
            path += "diving/0" + str(index) + ".avi"
        else:
            path += "diving/" + str(index) + ".avi"

        all_vids_preds.append(predict(path))

    vidsDf = pd.DataFrame(all_vids_preds, columns=(range(0, 103)))                                                      # each row in df is a list of 103 predictions for 1 video
    scores = load_data()['score'].tolist()                                                                              # append y column
    vidsDf['score'] = scores
    vidsDf.to_csv(path+'/new_data.csv')


def pred_splited_vids():                                                                                                # Now, create .csv file to the predictions made by the seperated models

    all_vids_preds = list()
    for index in range(1, 371):
        path = '/Users/I555250/PycharmProjects/olympicVAR/'
        if index < 10:
            path += "diving/00" + str(index) + ".avi"
        elif index < 100:
            path += "diving/0" + str(index) + ".avi"
        else:
            path += "diving/" + str(index) + ".avi"

        all_vids_preds.append(predict_by_parts(path))

    vidsDf = pd.DataFrame(all_vids_preds, columns=(range(0, 103)))
    scores = load_data()['score'].tolist()
    vidsDf['score'] = scores
    vidsDf.to_csv(path+'/new_splited_data.csv')



def calc_all_vids_somersaults():                                                                                        # Adding another feature to the data - amount of somersaults

    all_vids_counter = list()
    for index in range(1, 371):
        path = ''
        if index < 10:
            path += "diving/00" + str(index) + ".avi"
        elif index < 100:
            path += "diving/0" + str(index) + ".avi"
        else:
            path += "diving/" + str(index) + ".avi"

        all_vids_counter.append(SomersaultsCounter.calc_somersaults(drawTrainersSkeleton(path)))

    df = pd.read_csv(path+'/new_data.csv')
    df['somer'] = all_vids_counter
    df.to_csv(path+'/new_data.csv')

    df = pd.read_csv(path + '/new_splited_data.csv')
    df['somer'] = all_vids_counter
    df.to_csv(path + '/new_splited_data.csv')

