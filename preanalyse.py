# auteur: Michel
# date:2022/6/25 15:13
# Michel est tres tres joli!!!!!! bon courage!!!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

inputfile = 'C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/Molecular.csv'
textdata = 'C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/Molecular_test.csv'
data = pd.read_csv(inputfile, encoding='gbk')
data_test = pd.read_csv(textdata, encoding='gbk')
data_fea = data.iloc[:, 1:]
data_featest = data_test.iloc[:, 1:]
print(data_fea)
index = []
# dataI = data_fea.iloc[:, 0].value_counts()
# print(dataI.get(dataI.keys()[0]))
for i in range(730):
    dataI = data_fea.iloc[:, i].value_counts()
    if dataI.get(dataI.keys()[0]) / 1974 > 0.9:
        index.append(i)

print(index)
indextest = []

# for i in range(730):
#     dataI = data_featest.iloc[:, i].value_counts()
#     if dataI.get(dataI.keys()[0]) / 50 > 0.9:
#         indextest.append(i)

print(indextest)
index = [0, 10, 15, 17, 18, 19, 50, 51, 54, 55, 63, 64, 68, 69, 107, 120, 121, 122, 124, 125, 126, 127, 128, 129, 132,
         137, 138, 139, 140, 141, 142, 143, 144, 146, 148, 152, 153, 158, 159, 160, 161, 163, 164, 165, 166, 169, 170,
         171, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198,
         199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
         221, 222, 223, 224, 226, 239, 240, 241, 243, 244, 245, 246, 247, 248, 251, 256, 257, 258, 259, 260, 261, 262,
         263, 265, 267, 271, 272, 277, 278, 279, 280, 282, 283, 284, 285, 288, 289, 290, 295, 296, 298, 299, 300, 301,
         302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
         325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 345, 358, 359,
         360, 362, 363, 364, 365, 366, 367, 370, 375, 376, 377, 378, 379, 380, 381, 382, 384, 386, 390, 391, 396, 397,
         398, 399, 401, 402, 403, 404, 407, 408, 409, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427,
         428, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,
         451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 464, 477, 478, 479, 481, 482, 483, 484, 485, 486,
         489, 494, 495, 496, 497, 498, 499, 500, 501, 503, 505, 509, 510, 514, 515, 516, 517, 518, 520, 521, 522, 523,
         526, 527, 528, 533, 534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552,
         553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574,
         575, 576, 577, 578, 579, 580, 581, 596, 609, 663, 667, 668, 669, 681, 682, 685, 686, 687, 688, 689, 690, 691,
         693, 694, 695, 696, 697, 700, 701, 704, 707, 708, 711, 712]

# 获取清洗后的列值名
fields = data_fea.columns.values
# print(len(index))
for i in range(len(index)):
    data_fea = data_fea.drop(columns=[fields[index[i]]])


# 清洗后的数据 730 筛选后 得出363列
# print(data_fea)


def MaxMinNormaliztion(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def ZscoreNormalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


# 对数据进行最大最小标准化
data_Maxminscore = MaxMinNormaliztion(data_fea)
print(data_Maxminscore)
featureArray = data_Maxminscore.columns[0:361]
resArryay = data_Maxminscore.columns[362]
x = data_Maxminscore[featureArray]
y = data_Maxminscore[resArryay]

rf = RandomForestRegressor(n_estimators=141, max_depth=None,random_state=0)
rf.fit(x, y)
names = data_Maxminscore
importance = rf.feature_importances_

print("Features sorted by their score:")
zip_tmp = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
cnt = 1

list1 = []
list3 = []
list4 = []
for item in range(30):
    list1.append(zip_tmp[item])
    list3.append(list1[item][0])
    list4.append(list1[item][1])
print(list1)
print(list3)
print(range(len(list3)))

list2 = ['Power', 'Name']
dataN = data_Maxminscore[list4]
plt.title("Feature Importance")

# 指定x坐标轴，高度等参数
plt.bar(x=range(30), height=list3, width=0.3, color='lightblue', align='center')
# 对x坐标轴的标签进行覆盖,rotation是角度
plt.xticks(range(len(list3)), list4, rotation='vertical')
# 限制x轴数据的大小
plt.xlim([-1, len(list1)])
plt.tight_layout()
plt.show()

outData = pd.DataFrame(columns=list2, data=list1)
Outdonne = pd.DataFrame(columns=list1, data=dataN)
outData.to_csv('C:/Users/MICHEL/Desktop/Licence3-2/machine learning/lab/power.csv', encoding="utf-8")

# print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))

# n_estimators = [int(x) for x in np.linspace(start = 10, stop = 300, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# bootstrap = [True, False]
# min_samples_leaf = [1, 2, 4]
# rf = RandomForestRegressor(random_state = 0)
# from sklearn.model_selection import RandomizedSearchCV
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)# Fit the random search model
#
# rf_random.fit(x_train,y_train)
#
#
# #print the best score throughout the grid search
# print(rf_random.best_score_)
# #print the best parameter used for the highest score of the model.
# print(rf_random.best_params_)
