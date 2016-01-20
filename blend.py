__author__ = 'arda'

import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import itertools
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import operator
from scipy import stats
import glob

folder = "/home/ubuntu/springleaf/"
data_folder = '/home/arda/Documents/kaggle-data/springleaf/'
l = glob.glob(folder + "xgb*.csv")

l = [val for val in l if 'FI' not in val]


pd_data = pd.DataFrame.from_csv(l[0])
pd_data.columns = [l[0].split('_')[2].split('.')[0]]


for file in l[1:]:
    col_name = file.split('_')[2].split('.')[0]; print col_name
    pd_data[col_name] = pd.DataFrame.from_csv(file)


# cols = [col for col in pd_data if int(col) >= 79690]


#harmonic mean
# preds = pd_data[cols].apply(lambda r: stats.hmean(r.values), 1)



#geometric mean
# preds = pd_data[cols].apply(lambda r: stats.mstats.gmean(r), 1)


preds = pd_data.mean(1)


preds = pd.DataFrame({"ID": pd_data.index, "target": preds})
preds.to_csv(folder + "blend_gmean.csv", index=None)