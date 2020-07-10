#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import copy
import math
import pickle
import time

#Transet
def readTrainSet(trainsetFile):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_table(trainsetFile, sep=',',names=names)
    user_list = list(df.user_id.unique())
    user_list.sort(reverse=True)
    user_maxnum = user_list[0]
    item_list = list(df.item_id.unique())
    item_list.sort(reverse=True)
    item_maxnum = item_list[0]
    TrairSetRatings = np.zeros((user_maxnum+1, item_maxnum+1))
    for row in df.itertuples():
        TrairSetRatings[row[1]][row[2]] = row[3]
    return TrairSetRatings
#Testset
def readTestSet(testsetFile,TrainSetRatings):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_table(testsetFile, sep=',',names=names)
    rows = len(TrainSetRatings)
    columns = len(TrainSetRatings[0])
    TestSetRatings = np.zeros((rows,columns))
    for row in df.itertuples():
        TestSetRatings[row[1]][row[2]] = row[3]
    return TestSetRatings
#Initialize PQ matrix
def initPQ(TrainSetRatings,n):
    P = []
    Q = []
    for u in range(len(TrainSetRatings)):
        P.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n + 1)]))
    for i in range(len(TrainSetRatings[0])):
        Q.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n + 1)]))
    P = np.array(P)
    Q = np.array(Q)
    return P,Q
#MAE and RMSE
def Error(Ratings,P,Q):
    maeError = []
    rmseError = []
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:
                continue
            hat_rui = np.clip(np.dot(P[u], Q[i]),1.,5.)
            error = math.fabs(Ratings[u][i] - hat_rui)
            maeError.append(error)
            rmseError.append(error ** 2)
    MAE = sum(maeError) / len(maeError)
    RMSE = math.sqrt(sum(rmseError) / len(rmseError))
    return MAE, RMSE
#Accuracy
def getF_value(Ratings,P,Q):
    actual_num = 0.0
    perfect_num = 0.0
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:
                continue
            else:
                actual_num += 1.0
                hat_rui = float(np.clip(np.dot(P[u], Q[i]), 1., 5.))
                if (round(Ratings[u][i]) == round(hat_rui)):
                    perfect_num += 1.0
    SPR = 1
    PPR = perfect_num / actual_num
    return SPR, PPR
#Predict rating：
def hatRui(u,i,P,Q,n):
    hat_rui = 0
    for k in range(n):
        hat_rui = hat_rui + P[u][k]*Q[i][k]
    return hat_rui
#Objective function：
def function_reg(dataSet,P,Q):
    totalvalue = 0
    for u in range(len(dataSet)):
        for i in range(len(dataSet[u])):
            if dataSet[u][i] > 0:
                err = (dataSet[u][i] - np.dot(P[u], Q[i])) ** 2 + lamda * (np.dot(P[u],P[u]) + np.dot(Q[i],Q[i]))
                totalvalue = totalvalue + err
    return totalvalue
#SGD
def SGD(R,P,Q):
    itera = 0
    # step = 0.01
    curError = Error(TestSetRatings, P, Q)
    trainError = Error(TrainSetRatings, P, Q)
    # prefvalue = function_reg(TrainSetRatings, P, Q)
    pretestError = curError
    pretrainError = trainError
    preprecision = getF_value(TestSetRatings,P,Q)
    print('pre_train_rmse : {} ; pre_train_mae: {}'.format(pretrainError[0],pretrainError[1]))
    print('pre_test_rmse : {} ; pre_test_mae: {}'.format(pretrainError[0],pretrainError[1]))
    print('pre_spr : {} ; pre_ppr : {}'.format(preprecision[0],preprecision[1]))
    # print('pre_favlue:{}'.format(prefvalue))
    curErrorlist = [pretestError]
    trainErrorlist = [pretrainError]
    # fvaluelist = [prefvalue]
    nonzero_row, nonzero_column = R.nonzero()
    nonzero_num = len(nonzero_row)
    sample_arr = np.arange(nonzero_num)
    for iter in range(1,iter_array[-1]+1):
        np.random.shuffle(sample_arr)
        for idx in sample_arr:
            u = nonzero_row[idx]
            i = nonzero_column[idx]
            eui = R[u][i] - np.dot(P[u],Q[i])
            gradPu = 2 * (lamda * P[u] - eui * Q[i])
            gradQi = 2 * (lamda * Q[i] - eui * P[u])
            P[u] = P[u] - step * gradPu
            Q[i] = Q[i] - step * gradQi
        curError = Error(TestSetRatings, P, Q)
        trainError = Error(TrainSetRatings, P, Q)
        # fvalue = function_reg(TrainSetRatings, P, Q)
        precision = getF_value(TestSetRatings,P,Q)
        curErrorlist.append(curError[0])
        trainErrorlist.append(trainError[0])
        # fvaluelist.append(fvalue)
        itera = itera + 1
        if itera in iter_array: 
            print('iteration:{}'.format(itera))
            print('train_rmse : {} ; train_mae : {}'.format(trainError[0], trainError[1]))
            print('test_rmse : {} ; test_mae : {}'.format(curError[0], curError[1]))
            print('spr : {} ; ppr : {}'.format(precision[0], precision[1]))
            # print('favlue:{}'.format(fvalue))
            print(80 *'*')
    print("Completely")
    return curErrorlist, trainErrorlist
#Learning curve
def print_curve(propose_rmse,compare_rmse,iter_array):
    plt.plot(iter_array, compare_rmse,label='testrmse', linewidth=5)
    plt.plot(iter_array, propose_rmse,label='trainrmse', linewidth=5)
    plt.title('ml;n=10;reg=0.1')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.yticks(np.arange(0.7,3,0.2),fontsize=16)
    plt.xlabel('iterations', fontsize=10)
    plt.ylabel('RMSE', fontsize=10)
    plt.legend(loc='best', fontsize=20)
    plt.show()
def save_rmse(rmse,path):
    try:
        f_name = open(path,'wb')
        pickle.dump(rmse,f_name)
    finally:
        f_name.close()
#Main function
if __name__ == '__main__':
    time_start = time.time()
    # np.random.seed(0)
    trainsetFile = r"./trainset_100k.csv"
    testsetFile = r"./testset_100k.csv"
    TrainSetRatings = readTrainSet(trainsetFile)
    TestSetRatings = readTestSet(testsetFile,TrainSetRatings)
    #Feature
    n = 10
    #Regularization parameter
    lamda = 0.1
    #Learning rate：
    step = 0.01
    #iteration
    iter_array = np.arange(20)
    P, Q = initPQ(TrainSetRatings, n)
    test_RMSE,train_RMSE = SGD(TrainSetRatings,P,Q)
    time_end = time.time()
    print("time:", time_end - time_start)
    # print_curve(test_RMSE[1:], train_RMSE[1:], iter_array)


