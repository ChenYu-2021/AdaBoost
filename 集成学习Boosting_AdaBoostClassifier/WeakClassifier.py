import numpy as np

class WeakClsassifier:
    # 单层决策树
    def cal_dic(self, X):
        ret_gt = {}
        for i in range(X.shape[1]): # 特征数，1. 特征选择
            ret_gt[i] = []
            for j in range(X.shape[0]):
                temp_threshold = X[j, i] # 2.选择第i个特征的值作为阈值
                temp_line = []
                for k in range(X.shape[0]): # 每一个样本
                    if X[k, i] >= temp_threshold: # 每一个样本的第j个特征的值是否大于阈值
                        temp_line.append(1) # 大于阈值，则是正类
                    else:
                        temp_line.append(-1) # 否则是反类
                ret_gt[i].append(temp_line)

        ret_lt = {}
        for i in range(X.shape[1]):  # 特征
            ret_lt[i] = []
            for j in range(X.shape[0]): # 样本
                temp_threshold = X[j, i] # 选定阈值
                temp_line = []
                for k in range(X.shape[0]):
                    if X[k, i] <= temp_threshold:
                        temp_line.append(1)
                    else:
                        temp_line.append(-1)
                ret_lt[i].append(temp_line)
        ret = {}
        ret['gt'] = ret_gt
        ret['lt'] = ret_lt
        return ret

    # calculate error for one dimension array
    def cal_error_line(self, y, line):
        ret = 0
        for i in range(len(y)): # 样本数
            if y[i] != line[i]:
                ret += self.W[i]
        return ret

    # calculate error for two dimension array
    def cal_error_lines(self, y, lines):
        ret = []
        for i in lines:
            ret.append(self.cal_error_line(y, i)) # self.cal_error_line(y, i)表示第一个样本的误差
        return ret
    def cal_error_dic(self, y, dic):
        ret_gt = {}
        for i in dic['gt']:
            ret_gt[i] = self.cal_error_lines(y, dic['gt'][i]) # i表示特征
        ret_lt = {}
        for i in dic['lt']:
            ret_lt[i] = self.cal_error_lines(y, dic['lt'][i])
        ret = {}
        ret['gt'] = ret_gt
        ret['lt'] = ret_lt
        return ret

    # select min error for error_dic
    def cal_error_min(self, error_dic):
        ret = 100000
        for key in error_dic:
            for i in error_dic[key]:
                temp = min(error_dic[key][i])
                if ret > temp:
                    ret = temp
        for key in error_dic:
            for i in error_dic[key]:
                if ret == min(error_dic[key][i]):
                    return ret, key, i, error_dic[key][i].index(ret)
    # train
    def fit(self, X, y, W):
        self.W = W
        # 通过单层决策树对样本数据X进行分类 得到的是2x36x36的字典，第一个表示第几个特征，第二个表示按照第几个样本的特征的值作为阈值，
        # 第三个表示样本分类的结果
        dic = self.cal_dic(X)
        error_dic = self.cal_error_dic(y, dic) # 得到每一个样本的误差，是一个字典，分为gt和lt
        error_min, self.decision_key, self.decision_feature, error_min_i = self.cal_error_min(error_dic)
        self.decision_threshold = X[error_min_i, self.decision_feature]
        self.pred = dic[self.decision_key][self.decision_feature][error_min_i]
        return