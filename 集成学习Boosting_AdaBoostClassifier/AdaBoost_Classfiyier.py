""" 利用集成学习中的AdaBoost对数据集进行分类 """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from WeakClassifier import  *
from plot import *
from AdaBoost import  *
data = pd.read_csv('data.csv')

columns = ['feature_0', 'feature_1', 'label']
data.columns = columns
X = data.drop(['label'], axis=1).values
y = data['label'].values
'''
weak = WeakClsassifier()
W = np.array([1] * len(y)) / len(y)
W.reshape(len(y), 1)
weak.W = W
dic = weak.cal_dic(X)
error = weak.cal_error_dic(y, dic)
print(error['gt'][0])
'''
# 训练AdaBoost分类器
clf = AdaBoostClassifier()
times = clf.fit(X, y)

# 画原始图
Plot2D(data).pause(3)

for i in range(times):
	if clf.weak[i].decision_feature==0:
		plt.plot([clf.weak[i].decision_threshold,clf.weak[i].decision_threshold],[0,8])
	else:
		plt.plot([0,8],[clf.weak[i].decision_threshold,clf.weak[i].decision_threshold])
plt.pause(10)

