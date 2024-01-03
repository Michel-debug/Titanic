# auteur: Michel
# date:2022/6/28 17:25
# Michel est tres tres joli!!!!!! bon courage!!!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from boto import sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from wordcloud import WordCloud



data = pd.read_csv('C:/Users/MICHEL/Desktop/Licence3-2/datamining/lab/titanic_data.csv')
# data.info()
# 删除对于聚类无关的列
data=data.drop(["Name", "Cabin", "Ticket", "PassengerId"],axis=1)
# data.info()
data["Age"]=data["Age"].fillna(data["Age"].mean())
# data.info()
data['Fare']=data['Fare'].fillna(data['Fare'].mean())
# data.info()
# 去除缺失行
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode())
# data.info()
data=data.dropna(axis=0,how="any")
data.info()
data['Sex'] = data['Sex'].map({'male' : 1, 'female' : 0}).astype(int)
# data.info()
# print(round(data['Age']))
# 四舍五入
data['Age']=round(data["Age"])
# 类型转换
data['Age']=list(map(int, data['Age']))
data['Embarked'] = data['Embarked'].map({'S' : 0, 'C' : 1 , 'Q': 2}).astype(int)
m=data[['Sex','Survived']].groupby(['Sex']).mean()
plt.bar([0,1],
[m.loc[0,'Survived'],m.loc[1,'Survived']],0.5,
color='blue',
alpha=0.9,
)
plt.xticks([0,1],['female','male'])
plt.show()
print(data)
data.info()
# k-means 聚类  由于船票价格与船票等级 相关联，所以我们采用船票等级 来代替带票价格 将Embarked 重新数值化
# print(data['Embarked'].value_counts()) 统计 结果 S 914  C 270  Q 123

zscore = preprocessing.StandardScaler()
data_zs = zscore.fit_transform(data)
SSE=[]
for k in range(1,9):
    estimator = KMeans(n_clusters=k)
    estimator.fit(data[['Survived', 'Pclass','Sex','Age','SibSp','Parch','Embarked']])
    SSE.append((estimator.inertia_))
X=range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()

# df_features = np.array(data)
# print(df_features)
# KMeans=KMeans(n_clusters=4)
# KMeans.fit(df_features)
# print(KMeans.labels_)
# plt.figure(figsize=(8,15))
# colors=['b', 'g', 'r', 'y']
# markers = ['o', 's', 'p', 'H']
# # print(df_features[0][0])
# Michel = 1
# for x in range(1,8) :
#     for y in range(1,8):
#         plt.subplot(7,7,Michel)
#         Michel += 1
#         for i,l in enumerate(KMeans.labels_):
#             plt.plot(df_features[i][y-1],df_features[i][x-1], color=colors[l], marker=markers[l], alpha=0.4)
# plt.show()

