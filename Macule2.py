import csv

import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import metrics

dataSet = pd.read_csv('C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/Molecular3.csv', header=0)
data_ADMET = pd.read_csv('C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/ADMET.csv', header=0)

data_test = pd.read_csv('C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/Molecular_TEST.csv')
featureArray = dataSet.columns[1:]
resArryay = data_ADMET.columns[1:]
X = dataSet[featureArray]
y = data_ADMET[resArryay]

test_featureArray = data_test.columns[1:]
test_X = data_test[test_featureArray]

clf = XGBClassifier()
clf.fit(X, y)
test_predict = clf.predict(test_X)
print(test_predict)
f = open('C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/ADMET_test.csv','w',encoding='utf-8',newline="")
csv_write = csv.writer(f)
csv_write.writerow(['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN'])
for i in range(len(test_predict)):
    csv_write.writerow(test_predict[i])
