# auteur: Michel
# date:2022/6/29 11:01
# Michel est tres tres joli!!!!!! bon courage!!!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

from sklearn.datasets import load_wine

data = pd.read_csv('C:/Users/MICHEL/Desktop/Licence3-2/datamining/lab/titanic_data.csv')
# data.info()
# 删除对于聚类无关的列
data = data.drop(["Name", "Cabin", "Ticket", "PassengerId"], axis=1)
# data.info()
data["Age"] = data["Age"].fillna(data["Age"].mean())
# data.info()
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
# data.info()
# 去除缺失行
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode())
# data.info()
data = data.dropna(axis=0, how="any")
# data.info()
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}).astype(int)
# data.info()
# print(round(data['Age']))
# 四舍五入
data['Age'] = round(data["Age"])
# 类型转换
data['Age'] = list(map(int, data['Age']))
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
print(data)


# wine = load_wine()
# data = wine.data  # 数据
# lables = wine.target  # 标签
# feaures = wine.feature_names
# df = pd.DataFrame(data, columns=feaures)
# print(df)

def standareData(df):
    """
    df : 原始数据
    return : data 标准化的数据
    """

    data2 = pd.DataFrame(index=df.index)  # 列名，一个新的dataframe
    columns = df.columns.tolist()  # 将列名提取出来
    for col in columns:
        d = df[col]
        max = d.max()
        min = d.min()
        mean = d.mean()
        data2[col] = ((d - mean) / (max - min)).tolist()

    return data2


# 某一列当做参照序列，其他为对比序列
def graOne(Data, m=0):
    """
    return:
    """
    print('Data', Data)
    columns = Data.columns.tolist()  # 将列名提取出来
    print(columns)
    # 第一步：无量纲化
    data3 = standareData(Data)
    # print('data3', data3)
    referenceSeq = data3.iloc[:, m]  # 参考序列
    data3.drop(columns[m], axis=1, inplace=True)  # 删除参考列
    compareSeq = data3.iloc[:, 0:]  # 对比序列
    row, col = compareSeq.shape
    # 第二步：参考序列 - 对比序列
    data_sub = np.zeros([row, col])
    print(col)
    for i in range(col):
        for j in range(row):
            data_sub[j, i] = abs(referenceSeq[j] - compareSeq.iloc[j, i])
    # 找出最大值和最小值
    maxVal = np.max(data_sub)
    minVal = np.min(data_sub)
    cisi = np.zeros([row, col])

    for i in range(row):
        for j in range(col):
            cisi[i, j] = (minVal + 0.5 * maxVal) / (data_sub[i, j] + 0.5 * maxVal)
    # 第三步：计算关联度
    result = [np.mean(cisi[:, i]) for i in range(col)]
    result.insert(m, 1)  # 参照列为1

    return pd.DataFrame(result)


def GRA(Data):
    df = Data.copy()
    columns = [str(s) for s in df.columns if s not in [None]]  # [1 2 ,,,12]
    # print(columns)
    df_local = pd.DataFrame(columns=columns)
    df.columns = columns
    for i in range(len(df.columns)):  # 每一列都做参照序列，求关联系数
        print(i)
        df_local.iloc[:, i] = graOne(df, m=i)[0]
    df_local.index = columns
    return df_local


# 热力图展示
def ShowGRAHeatMap(DataFrame):
    colormap = plt.cm.hsv
    ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set_title('Wine GRA')
    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(DataFrame)
    mask[np.triu_indices_from(mask)] = True  # np.triu_indices 上三角矩阵

    with sns.axes_style("white"):
        sns.heatmap(DataFrame,
                    cmap="YlGnBu",
                    annot=True,
                    mask=mask,
                    )
    # plt.show()


data_wine_gra = GRA(data)
ShowGRAHeatMap(data_wine_gra)
