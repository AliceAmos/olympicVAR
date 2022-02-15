import pandas as pd
from predictVid import *
from preprocessing import *

SAMPLE = '/Users/I555250/PycharmProjects/olympicVAR/004.avi'

def pred_vids():

    all_vids_preds = list()
    for index in range(2, 371):
        path = '/Users/I555250/PycharmProjects/olympicVAR/'
        if index < 10:
            path += "diving/00" + str(index) + ".avi"
        elif index < 100:
            path += "diving/0" + str(index) + ".avi"
        else:
            path += "diving/" + str(index) + ".avi"

        all_vids_preds.append(predict(path))
        # row_preds = pd.Series(predict(path))
        # row_df = pd.DataFrame([row_preds])
        # vidsDf = pd.concat([row_df, vidsDf], ignore_index=True)

    vidsDf = pd.DataFrame(all_vids_preds, columns=(range(0, 103)))
    print(vidsDf)
    vidsDf.to_csv('/Users/I555250/PycharmProjects/olympicVAR/new_data.csv')
