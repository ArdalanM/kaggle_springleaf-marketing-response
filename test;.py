__author__ = 'arda'


import xgboost as xgb
import numpy as np
import pandas as pd

size = int(1e5)
num_feat = 10
X = np.vstack((np.random.randint(1,10,(size,num_feat)),np.random.randint(2,20,(size,num_feat))))


pd_data = pd.DataFrame(X)
#pd_data['y'] = Y
pd_data['lol'] = np.zeros((2*size))
pd_data['xd'] = np.zeros((2*size))

pd_data.columns = ['A','B','c','D','E','F','G','H','T','R','x','xdlol' ]

X = np.array(pd_data)
Y = np.hstack((np.zeros((size)),np.ones((size))))


params = {}
params["objective"] = "binary:logistic"
params["eta"] = 1.
params["eval_metric"] = 'error'
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = .5
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 9
params["alpha"] = 1

plst = list(params.items())

#Using 5000 rows for early stopping.
offset = 20000
num_rounds = 10

#create a train and validation dmatrices
xgtrain = xgb.DMatrix(X[offset:, :], label=Y[offset:], missing=np.nan)
xgval = xgb.DMatrix(X[:offset,:], label=Y[:offset], missing=np.nan)

#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=200)


def CreateDataFrameFeatureImportance(model, pd_data):
    dic_fi = model.get_fscore()
    df = pd.DataFrame(dic_fi.items())
    df.columns = ['features', 'score']
    df['col_indice'] = df['features'].apply(lambda r: r.replace('f','')).astype(int)
    df['feat_name'] = df['col_indice'].apply(lambda r: pd_data.columns[r])
    return df.sort('score', ascending=False)



df = CreateDataFrameFeatureImportance(model, pd_data)