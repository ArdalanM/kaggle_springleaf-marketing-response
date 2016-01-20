from nis import cat

__author__ = 'arda'


"""
Springleaf
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import itertools
import time
from sklearn.preprocessing import LabelEncoder
import datetime
import operator



# data_folder = '/home/arda/Documents/kaggle-data/springleaf/'
data_folder = '/home/ubuntu/springleaf/'

def xgboost_pred(train,labels,test, params, num_rounds):

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


	plst = list(params.items())

	#Using 5000 rows for early stopping.
	offset = 20000

	# num_rounds = 2
	xgtest = xgb.DMatrix(X_test, missing=np.nan)

	#create a train and validation dmatrices
	xgtrain = xgb.DMatrix(X[offset:, :], label=Y[offset:], missing=np.nan)
	xgval = xgb.DMatrix(X[:offset,:], label=Y[:offset], missing=np.nan)

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
	# model = xgb.train(plst, xgtrain, watchlist, early_stopping_rounds=120)
	y_pred = model.predict(xgtest,ntree_limit=model.best_iteration)

	return y_pred, model, params
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

#zipcode = 'VAR_0241' and 'VAR_0274', zipcode_and_code = 'VAR_0212' ,state_char = 'VAR_0237'



A = pd_train.copy()
B = pd_test.copy()


pd_train = A.copy()
pd_test = B.copy()


######################
#one hot state cols##
######################
# print("One hot other cols...")
# some_other_cat_cols = ['VAR_0005', 'VAR_1934']



cols = list(set(cat_cols + ['VAR_0241','VAR_0212', 'VAR_0237', 'VAR_0274', 'VAR_0200']))
pd_train, pd_test = pd_train[cols].reset_index(drop=1), pd_test[cols].reset_index(drop=1)



print("Extracting 2 digits zip code...")
pd_train['VAR_0241_2'] = pd_train['VAR_0241'].fillna(-1).apply(lambda r: int(str(r)[:2]))
pd_test['VAR_0241_2'] = pd_test['VAR_0241'].fillna(-1).apply(lambda r: int(str(r)[:2]))


#Adding count column
for col in ['VAR_0241', 'VAR_0237', 'VAR_0274', 'VAR_0200', 'VAR_0342', 'VAR_0325', 'VAR_0354', 'VAR_0353']:
	col_name = col + "_ctn"
	pd_train[col_name] = pd_train.fillna(-1).groupby(col)[col].transform('count')
	pd_test[col_name] = pd_test.fillna(-1).groupby(col)[col].transform('count')



for col in ['VAR_0237', 'VAR_0274', 'VAR_0200', 'VAR_0342', 'VAR_0325', 'VAR_0354', 'VAR_0353']:
	count_col = col + "_ctn"
	most_freq = pd_train[[col, count_col]].groupby(col).agg(lambda r: np.mean(r)).sort(count_col, ascending=False).index[:5].values

	for val in most_freq:
		pd_train[col + '_' + val] = pd_train[col].apply(lambda r: 1 if r == val else 0)
	pd_train[col + "_other"] = pd_train[col].apply(lambda r: 1 if r not in most_freq else 0)





######################
char_cols =  [col for col in pd_train.columns if pd_train[col].dtype==object]


pd_train, pd_test = LabelEncodeColumns(pd_train, pd_test, char_cols)


cols = pd_train.columns
# cols = ['VAR_0214', 'VAR_0200', 'VAR_0342', 'VAR_0274', 'VAR_0237',
# 		'VAR_0354', 'VAR_0325', 'VAR_0404', 'VAR_0352', 'VAR_0353',
# 		'VAR_0005', 'VAR_1934', 'VAR_0493', 'VAR_0305', 'VAR_0283',
# 		'VAR_0001', 'VAR_0467', 'VAR_0466', 'VAR_0232', 'VAR_0226',
# 		'VAR_0230', 'VAR_0236', 'VAR_0202']

X = np.array(pd_train)
X_test = np.array(pd_test)




#################
###FIT MODEL#####
#################
l_eta = [0.03]
l_max_depth = [15]
l_alpha=[2]
l_colsample_bytree = [0.5]

for eta in l_eta:
	for max_depth in l_max_depth:
		for alpha in l_alpha:
			for colsample_bytree in l_colsample_bytree:

				num_round = 400
				params = {}
				params["objective"] = "binary:logistic"
				# params["eta"] = 0.05
				params["eta"] = eta
				params["eval_metric"] = 'auc'
				# params["min_child_weight"] = 6
				params["min_child_weight"] = 2
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
				y_pred, model, params = xgboost_pred(X,Y, X_test, num_rounds=num_round, params=params)
				print("Elapsed time %i sec" % (time.time() - starttime))
				filename = "xgb"+str(num_round)+"_"+str(model.best_score).split(".")[1] ; print model.best_score
			# #
			# #dump models
			# model.save_model(data_folder + filename+".m")
			# pickle.dump(params, open(data_folder + filename+".param", 'wb'))
			# #
			# #generate solution
			# preds = pd.DataFrame({"ID": test_idx, "target": y_pred})
			# preds.to_csv(data_folder + filename +".csv", index=None)





df = CreateDataFrameFeatureImportance(model, pd_train)





# print("Deleting column with unique feature")
# unique_cols =  [col for col in pd_train.columns if len(pd_train[col].unique())==1]
# pd_train.drop(unique_cols +['ID', 'target'], axis = 1, inplace=True)
# pd_test.drop(unique_cols + ['ID'], axis = 1, inplace=True)





