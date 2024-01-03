# auteur: Michel
# date:2022/6/25 13:15
# Michel est tres tres joli!!!!!! bon courage!!!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

inputfile = 'C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/Molecular_30feature.csv'
inputfile_test = 'C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/Molecular_TEST_30feature.csv'
data = pd.read_csv(inputfile, encoding='gbk')
data_test = pd.read_csv(inputfile_test, encoding='gbk')


def maxminnormaliztion(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


print(data)
print(data_test)
featureArray = data.columns[1:]
resArryay = data.columns[0]
# X = maxminnormaliztion(data[featureArray])
# y = data[resArryay]
X = preprocessing.scale(data[featureArray])
y = data[resArryay]

test_featureArray = data_test.columns[1:]
test_resArryay = data_test.columns[0]
test_X = preprocessing.scale(data_test[test_featureArray])
test_y = data_test[test_resArryay]

n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
bootstrap = [True, False]
min_samples_leaf = [1, 2, 4]
rf = RandomForestRegressor(random_state=0)

# random_grid = {'n_estimators': n_estimators,
#                 'max_features': max_features,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'min_samples_leaf': min_samples_leaf,
#                 'bootstrap': bootstrap}
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
#                                 random_state=0, n_jobs=-1)  # Fit the random search model
#
# rf_random.fit(X, y)
#
# # print the best score throughout the grid search
# print(rf_random.best_score_)
# # print the best parameter used for the highest score of the model.
# print(rf_random.best_params_)
regressor = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_split=2, min_samples_leaf=2,
                                  max_features='sqrt', max_depth=100, bootstrap=False)
regressor.fit(X, y)
y_pred = regressor.predict(test_X)
score = metrics.mean_squared_error(test_y, y_pred)
print(y_pred)
print(score)
plt.plot(y_pred, 'b', alpha=0.5, linewidth=2, label='predicte')
plt.plot(test_y, 'r', alpha=0.5, linewidth=2, label='true')
plt.legend()
plt.xlabel('number')
plt.ylabel('activity coefficient')
plt.ylim(4,9)
plt.show()