# auteur: Michel
# date:2022/6/29 9:18
# Michel est tres tres joli!!!!!! bon courage!!!
# 查看聚类结果
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# data.info()
data['Sex'] = data['Sex'].map({'male' : 1, 'female' : 0}).astype(int)
# data.info()
# print(round(data['Age']))
# 四舍五入
data['Age']=round(data["Age"])
# 类型转换
data['Age']=list(map(int, data['Age']))
m=data[['Sex','Survived']].groupby(['Sex']).mean()
data.info()
# k-means 聚类  由于船票价格与船票等级 相关联，所以我们采用船票等级 来代替带票价格 将Embarked 重新数值化
# print(data['Embarked'].value_counts()) 统计 结果 S 914  C 270  Q 123
data['Embarked'] = data['Embarked'].map({'S' : 0, 'C' : 1 , 'Q': 2}).astype(int)
data.info()
zscore = preprocessing.StandardScaler()
data_zs = zscore.fit_transform(data)
# print(data_zs)
k=4
kmeans_model = KMeans(n_clusters=k,random_state=123)
fit_kmeans = kmeans_model.fit(data_zs)
kmeans_cc = kmeans_model.cluster_centers_  # 聚类中心
print('各类聚类中心为：\n',kmeans_cc)
kmeans_labels = kmeans_model.labels_  # 样本的类别标签
print('各样本的类别标签为：\n',kmeans_labels)
r1 = pd.Series(kmeans_model.labels_).value_counts()  # 统计不同类别样本的数目
print('最终每个类别的数目为：\n',r1)
#
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_,columns = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])   # 将聚类中心放在数据框中
cluster_center.index = pd.DataFrame(kmeans_model.labels_ ).drop_duplicates().iloc[:,0]  # 将样本类别作为数据框索引
print(cluster_center)


# 代码7-10

import matplotlib.pyplot as plt
# 客户分群雷达图
labels = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
legen = ['聚类' + str(i + 1) for i in cluster_center.index]  # 客户群命名，作为雷达图的图例
lstype = ['-','-','-','-']
kinds = list(cluster_center.iloc[:, 0])

# 由于雷达图要保证数据闭合，因此再添加首列，并转换为 np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['Survived']]], axis=1)
centers = np.array(cluster_center.iloc[:, 0:])
print(centers)
# 分割圆周长，并让其闭合
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
angle = np.concatenate((angle, [angle[0]]))

# 绘图
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, polar=True)  # 以极坐标的形式绘制图形
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 画线
for i in range(len(kinds)):
    print(centers[i])
    ax.plot(angle, centers[i], linestyle=lstype[i], linewidth=2, label=kinds[i])
# 添加属性标签
plt.thetagrids(range(0, 360, int(360/len(labels))), (labels))
plt.title('泰塔尼克号聚类分析雷达图')
plt.legend(legen)
plt.show()
plt.close

scores = []
for i in range(2, 10):
    km = KMeans(        n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0      )
    km.fit(data_zs)
    scores.append(metrics.silhouette_score(data_zs, km.labels_ , metric='euclidean'))
plt.plot(range(2,10), scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()
plt.close()

