__author__ = 'arda'

import pandas as pd
import zipfile
import pickle



# folder = "/home/arda/Documents/kaggle-data/springleaf/"
folder = '/home/ubuntu/springleaf/'


def loadFileinZipFile(zip_filename, filename):
    with zipfile.ZipFile(zip_filename, 'r') as myzip:
        pd_data= pd.read_csv(myzip.open(filename))
    return pd_data

def splitDatetime(x):
    return x.year, x.month, x.dayofweek, x.hour

def parseDate(pd_train):

    #datetime column
    date_cols = ['VAR_0073', 'VAR_0075','VAR_0156','VAR_0157',
         'VAR_0158','VAR_0159','VAR_0166','VAR_0167',
         'VAR_0168','VAR_0169','VAR_0176','VAR_0177',
         'VAR_0178','VAR_0179','VAR_0204','VAR_0217']

    #creating new df
    for col in date_cols:

        print col
        pd_temp = pd.DataFrame()

        #parsing string into datetime column
        # datetime_col = pd.to_datetime(pd_train[col], format='%d%b%y:%H:%M:%S')
        pd_train[col] = pd.to_datetime(pd_train[col], format='%d%b%y:%H:%M:%S')

        #split datetime column into multiple column
        pd_temp[col+'_year'], pd_temp[col+'_month'], pd_temp[col+'_DoW'], pd_temp[col+'_hour'] = zip(*pd_train[col].map(splitDatetime))

        #Creating new DataFrame
        pd_train = pd.concat([pd_train, pd_temp], axis=1)

        #del pd_train[col]
    return pd_train


l = ['train', 'test']


for filename in l:

    print(filename)
    print("Loading ...")
    pd_data = loadFileinZipFile(folder + filename + ".csv.zip", filename + ".csv")

    print("Parsing Dates...")
    pd_data = parseDate(pd_data)

    print("Pickling...")
    pd_data.to_pickle(folder + filename + "_parsed.p")

    del pd_data



