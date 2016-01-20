#!/usr/bin/python
__author__ = 'arda'
"""Springleaf Kaggle challenge """

import pickle
import itertools
import time
import datetime
import operator

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation

# data_folder = '/home/arda/Documents/kaggle-data/springleaf/'
data_folder = ''
# data_folder = '/home/ubuntu/springleaf/'

def addDateCombinaisons(pd_data, date_cols):

    combinaisons = itertools.combinations(date_cols, 2)
    for var1, var2 in combinaisons:
        # print var1, var2

        col_name = "delta_" + var1 + "_" + var2
        # print col_name
        diff = (pd_data[var1] - pd_data[var2]).astype('timedelta64[m]')
        pd_data[col_name] = diff

    return pd_data

def createDummiesFromColumns(pd_data, cols):
	return pd.get_dummies(pd_data[cols]).astype('int16')

def CreateDataFrameFeatureImportance(model, pd_data):
    dic_fi = model.get_fscore()
    df = pd.DataFrame(dic_fi.items())
    df.columns = ['features', 'score']
    df['col_indice'] = df['features'].apply(lambda r: r.replace('f','')).astype(int)
    df['feat_name'] = df['col_indice'].apply(lambda r: pd_data.columns[r])
    return df.sort('score', ascending=False)

def LabelEncodeColumns(pd_train, pd_test, cols):
	#########################
	#Label encode char cols##
	#########################
	le = LabelEncoder()

	for col in cols:
		vec = pd_train[col].append(pd_test[col]).values
		y = le.fit_transform(vec)

		pd_train[col] = y[:len(pd_train)]
		pd_test[col] = y[len(pd_train):]

	return pd_train, pd_test

def xgboost_pred(x_train,y_train,x_val, y_val, params, num_rounds):

	# params = {}
	# params["objective"] = "binary:logistic"
	# params["eta"] = 0.05
	# params["eval_metric"] = 'auc'
	# params["min_child_weight"] = 6
	# params["subsample"] = 0.7
	# params["colsample_bytree"] = 0.5
	# params["scale_pos_weight"] = 1
	# params["silent"] = 1
	# params["max_depth"] = 9
	# # params["nthreads"] = 3
	# params["alpha"] = 4
	# # params["num_round"] = 2


	# offset = 20000

	#create a train and validation dmatrices
	xgtrain = xgb.DMatrix(x_train, label=y_train, missing=np.nan)
	xgval = xgb.DMatrix(x_val, label=y_val, missing=np.nan)

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=True)

	return model, params




print("Load data...")
pd_train = pd.read_pickle(data_folder + 'train_parsed.p')
pd_test = pd.read_pickle(data_folder + 'test_parsed.p')
Y = pd_train.target.values
test_idx = pd_test.ID



print("Deleting column with unique feature")
unique_cols = [col for col in pd_train.columns if pd_train[col].value_counts().shape[0] <= 1]
pd_train.drop(unique_cols +['ID', 'target'], axis = 1, inplace=True)
pd_test.drop(unique_cols + ['ID'], axis = 1, inplace=True)


print('Getting col types in lists....')
cat_cols = []
num_cols = []
date_cols = []
for col in pd_train.columns:
	one_col = pd_train[col]
	if one_col.dtype in ['float64', 'int64']:
		num_cols.append(col)
	elif one_col.dtype == 'object':
		cat_cols.append(col)
	else:
		date_cols.append(col)




#########################################
#DEALING WITH CATEGORICAL VARIABLES#####
#########################################
print("Extracting 2 digits zip code...")
pd_train['VAR_0241_2'] = pd_train['VAR_0241'].fillna(-1).apply(lambda r: int(str(r)[:2]))
pd_test['VAR_0241_2'] = pd_test['VAR_0241'].fillna(-1).apply(lambda r: int(str(r)[:2]))


# print("Adding count column...")
# for col in ['VAR_0241', 'VAR_0237', 'VAR_0274', 'VAR_0200', 'VAR_0342', 'VAR_0325', 'VAR_0354', 'VAR_0353']:
# 	col_name = col + "_ctn"
# 	pd_train[col_name] = pd_train.fillna(-1).groupby(col)[col].transform('count')
# 	pd_test[col_name] = pd_test.fillna(-1).groupby(col)[col].transform('count')


# print("Adding most freq items...")
# for col in ['VAR_0237', 'VAR_0274', 'VAR_0200', 'VAR_0342', 'VAR_0325', 'VAR_0354', 'VAR_0353']:
# 	count_col = col + "_ctn"
# 	most_freq = pd_train[[col, count_col]].groupby(col).agg(lambda r: np.mean(r)).sort(count_col, ascending=False).index[:5].values
#
# 	for val in most_freq:
# 		pd_train[col + '_' + val] = pd_train[col].apply(lambda r: 1 if r == val else 0)
# 		pd_test[col + '_' + val] = pd_test[col].apply(lambda r: 1 if r == val else 0)
#
# 	pd_test[col + "_other"] = pd_test[col].apply(lambda r: 1 if r not in most_freq else 0)
# 	pd_test[col + "_other"] = pd_test[col].apply(lambda r: 1 if r not in most_freq else 0)




#########################################
#DEALing WIth DATES######################
#########################################
print("Adding Date combinaisons...")
pd_train = addDateCombinaisons(pd_train, date_cols)
pd_test = addDateCombinaisons(pd_test, date_cols)

print("Drop date columns...")
pd_train.drop(date_cols, axis = 1, inplace=True)
pd_test.drop(date_cols, axis = 1, inplace=True)



print("Label encode remaining columns...")
char_cols =  [col for col in pd_train.columns if pd_train[col].dtype==object]
pd_train, pd_test = LabelEncodeColumns(pd_train, pd_test, char_cols)







X = np.array(pd_train)
X_test = np.array(pd_test)




skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=5, test_size=0.2,random_state=123)


#################
###FIT MODEL#####
#################
l_eta = [0.01]
l_max_depth = [9]
l_alpha=[4]
l_colsample_bytree = [0.8]

for eta in l_eta:
	for max_depth in l_max_depth:
		for alpha in l_alpha:
			for colsample_bytree in l_colsample_bytree:


				for i, (tr_idx, te_idx) in enumerate(skf):

					x_train = X[tr_idx]
					x_val = X[te_idx]
					y_train = Y[tr_idx]
					y_val = Y[te_idx]

					num_round = 3
					params = {}
					params["objective"] = "binary:logistic"
					# params["eta"] = 0.05
					params["eta"] = eta
					params["eval_metric"] = 'auc'
					params["min_child_weight"] = 6
					params["subsample"] = 0.7
					# params["colsample_bytree"] = 0.5
					params["colsample_bytree"] = colsample_bytree
					params["scale_pos_weight"] = 1
					params["silent"] = 1
					# params["max_depth"] = 9
					params["max_depth"] = max_depth
					# params["nthreads"] = 3
					params["alpha"] = alpha
					# params["num_round"] = 2
					print("Fitting with following params: %s" % str(params))
					#
					#
					#fit model
					starttime = time.time()
					model, params = xgboost_pred(x_train, y_train, x_val, y_val, params, num_round)
					print("Elapsed time %i sec" % (time.time() - starttime))
					filename = "xgb"+str(num_round)+"_"+str(model.best_score).split(".")[1]+"_"+str(i)
					#
					#dump models
					model.save_model(data_folder + filename+".m")
					pickle.dump(params, open(data_folder + filename+".param", 'wb'))
					pd_data = CreateDataFrameFeatureImportance(model, pd_train)
					pd_data.to_csv(data_folder + filename + "_FI.csv")
					#
					#generate solution
					y_pred = model.predict(xgb.DMatrix(X_test, missing=np.nan),ntree_limit=model.best_iteration)
					preds = pd.DataFrame({"ID": test_idx, "target":y_pred})
					preds.to_csv(data_folder + filename +".csv", index=None)