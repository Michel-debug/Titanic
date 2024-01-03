# auteur: Michel
# date:2022/6/29 10:15
# Michel est tres tres joli!!!!!! bon courage!!!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;


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
print(data['Fare'])
plt.scatter(data['Age'],data['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()
data['Age']=list(map(int, data['Age']))
data['Embarked'] = data['Embarked'].map({'S' : 0, 'C' : 1 , 'Q': 2}).astype(int)
sns.set_style('ticks') # 十字叉
plt.axis('equal')       #行宽相同
data['Survived'].value_counts().plot.pie(autopct='%1.2f%%')

sns.set()
sns.set_style('ticks')
plt.figure(figsize=(12,5))
plt.subplot(121)
data['Age'].hist(bins=80)
plt.xlabel('Age')
plt.ylabel('Num')
plt.subplot(122)
data.boxplot(column='Age', showfliers=True) # 是否显示异常值

data['Age'].describe()
data[['Sex','Survived']].groupby('Sex').mean().plot.bar()

survive_sex=data.groupby(['Sex','Survived'])['Survived'].count()

fig,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot('Pclass','Age',hue='Survived',data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')

sns.violinplot('Sex','Age',hue='Survived',data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')

# 老人和小孩存活率
plt.figure(figsize=(18,4))
# 年龄都转换成整数

average_age=data[['Age','Survived']].groupby('Age',as_index=False).mean()

sns.barplot(x='Age',y='Survived',data=average_age,palette='BuPu')
plt.grid(linestyle='--',alpha=0.5)


# 票价与存活与否的关系
fig,ax=plt.subplots(1,2,figsize=(15,4))
data['Fare'].hist(bins=70,ax=ax[0])
data.boxplot(column='Fare',by='Pclass',showfliers=False,ax=ax[1])

fare_not_survived=data['Fare'][data['Survived']==0]
fare_survived=data['Fare'][data['Survived']==1]
# 筛选数据

average_fare=pd.DataFrame([fare_not_survived.mean(),fare_survived.mean()])
std_fare=pd.DataFrame([fare_not_survived.std(),fare_survived.std()])

average_fare.plot(yerr=std_fare,kind='bar',figsize=(15,4),grid=True)

# 关系热力图
corrMatt = data[['Survived', 'Pclass','Sex','Age','SibSp','Parch','Embarked']]
mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


