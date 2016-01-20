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
import operator


data_folder = '/home/arda/Documents/kaggle-data/springleaf/'
# data_folder = '/home/ubuntu/springleaf/'

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
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=200)
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


print("Load data...")
pd_train = pd.read_pickle(data_folder + 'train_parsed.p')
pd_test = pd.read_pickle(data_folder + 'test_parsed.p')
Y = pd_train.target.values
test_idx = pd_test.ID


print("Deleting column with unique feature")
unique_cols =  [col for col in pd_train.columns if len(pd_train[col].unique())==1]
pd_train.drop(unique_cols +['ID', 'target'], axis = 1, inplace=True)
pd_test.drop(unique_cols + ['ID'], axis = 1, inplace=True)


print("Adding Date combinaisons...")
date_cols = ['VAR_0073', 'VAR_0075','VAR_0156','VAR_0157',
     'VAR_0158','VAR_0159','VAR_0166','VAR_0167',
     'VAR_0168','VAR_0169','VAR_0176','VAR_0177',
     'VAR_0178','VAR_0179','VAR_0204','VAR_0217']
pd_train = addDateCombinaisons(pd_train, date_cols)
pd_test = addDateCombinaisons(pd_test, date_cols)


print("Drop date columns...")
pd_train.drop(date_cols, axis = 1, inplace=True)
pd_test.drop(date_cols, axis = 1, inplace=True)



######################
#one hot state cols##
######################
print("One hot dept cols...")
dept_cols = ['VAR_0274', 'VAR_0237']
pd_dummy_train = createDummiesFromColumns(pd_train, dept_cols)
pd_dummy_test = createDummiesFromColumns(pd_test, dept_cols)

for col in pd_dummy_train.columns:
	pd_train[col] = pd_dummy_train[col]

for col in pd_dummy_test.columns:
	pd_test[col] = pd_dummy_test[col]

# pd_train.drop(dept_cols, axis = 1, inplace=True)
# pd_test.drop(dept_cols, axis = 1, inplace=True)
del pd_dummy_train; del pd_dummy_test


######################
#one hot state cols##
######################
print("One hot other cols...")
# dept_cols = ['VAR_0274', 'VAR_0237']
some_other_cat_cols = ['VAR_0005', 'VAR_1934']
pd_dummy_train = createDummiesFromColumns(pd_train, some_other_cat_cols)
pd_dummy_test = createDummiesFromColumns(pd_test, some_other_cat_cols)

for col in pd_dummy_train.columns:
	pd_train[col] = pd_dummy_train[col]

for col in pd_dummy_test.columns:
	pd_test[col] = pd_dummy_test[col]

# pd_train.drop(some_other_cat_cols, axis = 1, inplace=True)
# pd_test.drop(some_other_cat_cols, axis = 1, inplace=True)
del pd_dummy_train; del pd_dummy_test



char_cols =  [col for col in pd_train.columns if pd_train[col].dtype==object]
num_cols =  [col for col in pd_train.columns if pd_train[col].dtype!=object]



#########################
#Label encode char cols##
#########################
le = LabelEncoder()

for col in char_cols:
	# full_data[col] = full_data[col].fillna(4242)
	vec = pd_train[col].append(pd_test[col]).values
	y = le.fit_transform(vec)

	pd_train[col] = y[:len(pd_train)]
	pd_test[col] = y[len(pd_train):]









filename = 'xgb3000_796871_79713'
m = xgb.Booster(model_file=data_folder + filename + '.m')
params = pickle.load(open(data_folder + filename+ '.param', 'rb'))

dic = m.get_fscore()

m.dump_model(fout='lol.m', with_stats=True)
m = xgb.Booster(model_file='lol.m')


df = pd.DataFrame(dic.items())
df[0] = df[0].apply(lambda r: r.replace('f','')).astype(int)
df = df.sort(0)






def get_FI(model, pd_train):
	importance = model.get_fscore()
	importance = sorted(importance.items(), key=operator.itemgetter(1))[::-1]

	df = pd.DataFrame(importance, columns=['feature', 'fscore'])
	df['feature'] = df['feature'].apply(lambda r: r.replace('f','')).astype(int)
	df['var'] = df['feature'].apply(lambda r: pd_train.columns[r])
	df['type'] = df['var'].apply(lambda r: pd_train[r].dtype.name )

	return df

importance = m.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))[::-1]

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['feature'] = df['feature'].apply(lambda r: r.replace('f','')).astype(int)
df['var'] = df['feature'].apply(lambda r: pd_train.columns[r])
df['type'] = df['var'].apply(lambda r: pd_train[r].dtype.name )


df = get_FI(m, pd_train)

# for col in pd_train.columns:
# 	l_unique_col = len(pd_train[col].unique())
# 	print l_unique_col


df[df['type']=='object']
test = pd_train[df[df['type']=='object']['var','fscore'].values]



[('f1412', 1),
 ('f1027', 1),
 ('f2117', 1),
 ('f2103', 1),
 ('f129', 1),
 ('f1925', 1),
 ('f1654', 1),
 ('f156', 1),
 ('f1939', 1),
 ('f2157', 1)]














# city_col = ['VAR_0200']
#one hot state cols
dept_cols = ['VAR_0274', 'VAR_0237']
a = pd.get_dummies(pd_train[dept_cols])


#one hot state cols
some_other_cat_cols = ['VAR_0005', 'VAR_1934']
pd.get_dummies(pd_train[some_other_cat_cols])



# for col in a:
# 	sum = np.sum(a[col])
# 	if sum < 2000:
# 		print sum


