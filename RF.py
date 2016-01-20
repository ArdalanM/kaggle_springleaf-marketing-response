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



# bad_feat = ['VAR_0478', 'VAR_0491', 'VAR_0672', 'VAR_0678', 'VAR_0670',
#        'VAR_0741', 'VAR_0374', 'delta_VAR_0157_VAR_0166', 'VAR_1433',
#        'VAR_1736', 'VAR_0259', 'VAR_0789', 'VAR_0533', 'VAR_1158',
#        'VAR_1630', 'delta_VAR_0158_VAR_0176', 'delta_VAR_0158_VAR_0204',
#        'VAR_1680', 'VAR_0809', 'VAR_1634', 'VAR_0995', 'VAR_0162',
#        'VAR_0157_month', 'VAR_0563', 'VAR_0565', 'VAR_0125',
#        'delta_VAR_0156_VAR_0157', 'VAR_0458', 'VAR_1932', 'VAR_1850',
#        'VAR_0470', 'VAR_1673', 'VAR_1672', 'VAR_0495', 'VAR_0679',
#        'VAR_0167_month', 'VAR_0276', 'VAR_0477', 'VAR_0430',
#        'VAR_0217_year', 'VAR_1597', 'VAR_1595', 'VAR_0551',
#        'delta_VAR_0166_VAR_0176', 'delta_VAR_0167_VAR_0168', 'VAR_1590',
#        'VAR_1586', 'delta_VAR_0159_VAR_0177', 'VAR_0177_year', 'VAR_0443',
#        'VAR_0373', 'VAR_0385', 'VAR_0185', 'delta_VAR_0158_VAR_0169',
#        'delta_VAR_0158_VAR_0168', 'delta_VAR_0158_VAR_0166', 'VAR_0047',
#        'VAR_0462', 'VAR_0152', 'VAR_0147', 'VAR_0149', 'VAR_0150',
#        'VAR_0092', 'delta_VAR_0157_VAR_0217', 'delta_VAR_0158_VAR_0159',
#        'delta_VAR_0157_VAR_0168', 'VAR_0163', 'VAR_1678', 'VAR_1688',
#        'VAR_1852', 'delta_VAR_0156_VAR_0177', 'VAR_0569', 'VAR_0568',
#        'VAR_0124', 'delta_VAR_0156_VAR_0167', 'VAR_0645', 'VAR_0453',
#        'VAR_1682', 'VAR_1050', 'VAR_1436', 'VAR_1168', 'VAR_1051',
#        'VAR_0184']
#
# pd_train.drop(bad_feat, axis = 1, inplace=True)
# pd_test.drop(bad_feat, axis = 1, inplace=True)

print("Remove columns with 2 features and nan...")
l = []
for col in pd_train.columns:
    unique = pd_train[col].unique()
    if len(unique)==2 and np.sum([1 for val in unique if str(val)=='nan']):
        l.append(col)
pd_train.drop(l, axis = 1, inplace=True)
pd_test.drop(l, axis = 1, inplace=True)

X = np.array(pd_train)
X_test = np.array(pd_test)




from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

offset = 20000

clf = RandomForestClassifier(n_estimators=2, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0., max_features='auto', max_leaf_nodes=None,
                             n_jobs=-1, class_weight='auto')



clf.fit(X,Y)












