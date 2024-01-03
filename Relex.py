# auteur: Michel
# date:2022/6/28 13:26
# Michel est tres tres joli!!!!!! bon courage!!!
import re

import PIL
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

data = pd.read_csv('C:/Users/MICHEL/Desktop/Licence3-2/datamining/lab/titanic_data.csv')

data_name_colume = data.columns[3]
data_name = data[data_name_colume]
print(data_name)
pd.set_option("display.max_columns", 10)
print(data.describe())


def getTitle(name):
    str1 = name.split(',')[1]  # Mr. Owen Harris
    str2 = str1.split('.')[0]  # Mr
    str3 = str1.split('.')[1]
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str4 = str3.strip()
    return str4
def getTitle_Nom(name):
    str0 = name.split(',')[0]
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str4 = str0.strip()
    return str4

# 存放提取后的特征

Nom_list=[]
Prenom_list=[]
for i in range(len(data_name)):

    Nom_list.append(getTitle_Nom(data_name[i]))
    Prenom_list.append(getTitle(data_name[i]))

# arr1=np.array(Nom_list)
# arr2=np.array(Prenom_list)
# arr3 = arr1 + arr2
# print(Nom_list)
# print(Prenom_list)

Nom_list=np.concatenate((Nom_list,Prenom_list),axis=0)

# print(Nom_list)
test_cut = ' '.join(Nom_list)
# print(len(Prenom_list))
# print(pd.Series(Nom_list).value_counts())
# print("===================+===============================")
# print(pd.Series(Prenom_list).value_counts())

# df = data_name.str.extract("(\w+,\s)\w+(.\s.+)")
# # print(df)
#
# n_name = df
# text_cut = jieba.lcut(str(n_name))
# text_cut = ' '.join(text_cut)
# print(n_name)

#根据名字 构造词云图

image1 = PIL.Image.open('C:/Users/MICHEL/Desktop/Licence3-2/datamining/lab/back.png')
MASK = np.array(image1)
word_cloud = WordCloud(font_path="simsun.ttc",
                       background_color="white",
                       mask=MASK,                # 指定词云的形状
                       )

word_cloud.generate(test_cut)
plt.subplots(figsize=(12,8))
plt.imshow(word_cloud)
plt.axis("off")
plt.show()
