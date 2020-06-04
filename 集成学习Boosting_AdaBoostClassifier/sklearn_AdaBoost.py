import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


# 用make_gaussian_quantiles生成多组多维的正态分布数据
# 生成2为正态分布，样本数为1000， 协方差为2
x1, y1 = make_gaussian_quantiles(cov=2, n_samples=200, n_features=2, n_classes=2, shuffle=True, random_state=1)
x2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, shuffle=True, random_state=1)


# 合并数据
'''
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    np.vstack((a,b))
    array([[1, 2, 3],
           [2, 3, 4]])
'''
X = np.vstack((x1, x2)) # 按行来合并数据
y = np.hstack((y1, 1 - y2)) # 按照列来合并数据

# 设定弱分类器为CART,且是单层的决策树
weakClassifier = DecisionTreeClassifier(max_depth=1)

# 构建模型
'''
scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，
SAMME使用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重
n_estimators: 最大的弱学习器的个数
learning_tate:每个弱学习器的权重缩减系数
'''
clf = AdaBoostClassifier(base_estimator=weakClassifier, algorithm='SAMME', n_estimators=300, learning_rate=0.8)
clf.fit(X, y)


# 绘制分类效果
x1_min = X[:, 0].min()-1
x1_max = X[:, 0].max()+1
x2_min = X[:, 1].min()-1
x2_max = X[:, 1].max()+1
'''
[X, Y] = meshgrid(x, y)
x的长度是m，y的长度是n，则得到的X和Y都是nm矩阵，X是按照x进行行复制，Y是按照y进行列复制
'''
x1_, x2_ = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
'''
ravel(a): 将多维数组降为1维
    x=np.array([[1,2],[3,4]])
    x.ravel()
    array([1, 2, 3, 4])
np.c_[a,b]:是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
'''
y_ = clf.predict(np.c_[x1_.ravel(), x2_.ravel()])
y_ = y_.reshape(x1_.shape)
'''
contourf(X, Y, Z):画等高线
X和Y是Z的坐标值
Z是绘制轮廓的高度，也叫函数值。在这里是经过AdaBoostClassifier后得到的值，函数是len(y)，列数是len(x)
'''
plt.contourf(x1_, x2_, y_, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()