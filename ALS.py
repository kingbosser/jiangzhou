#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import copy
import math
import pickle
import time
#Trainset
def readTrainSet(trainsetFile):
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_table(trainsetFile, sep=',', names=names)
    user_list = list(df.user_id.unique())
    user_list.sort(reverse=True)
    user_maxnum = user_list[0]
    item_list = list(df.item_id.unique()) 
    item_list.sort(reverse=True)
    item_maxnum = item_list[0]
    TrairSetRatings = np.zeros((int(user_maxnum)+1, int(item_maxnum)+1))
    for row in df.itertuples():
        TrairSetRatings[row[1]][row[2]] = row[3]
    return TrairSetRatings
#Testset
def readTestSet(testsetFile,TrainSetRatings):
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_table(testsetFile, sep= ',', names=names)
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
#RMSE
def Error(Ratings,P,Q,n):
    MAE_error = 0
    RMSE_error = 0
    count = 0
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:continue
            count = count + 1
            hat_rui = np.dot(P[u],Q[i])
            err = (Ratings[u][i] - hat_rui)
            MAE_error = MAE_error + math.fabs(err)
            RMSE_error = RMSE_error +err**2
    MAE = MAE_error / count
    RMSE = (RMSE_error / count) ** 0.5
    return MAE, RMSE

def rating_ability(Ratings,P,Q,n):
    perfect_num = 0.0
    count = 0
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:continue
            count = count + 1
            hat_rui = np.dot(P[u],Q[i])
            if Ratings[u][i] == round(hat_rui):
                perfect_num += 1
    PPR = perfect_num / count
    return PPR

#Predict ratings：
def hatRui(u,i,P,Q,n):
    hat_rui = 0
    for k in range(n):
        hat_rui = hat_rui + P[u][k]*Q[i][k]
    return hat_rui
#Objective function：
def function(dataSet,P,Q):
    u_list = []
    i_list = []
    err = 0
    for u in range(len(dataSet)):
        for i in range(len(dataSet[u])):
            if dataSet[u][i] > 0:
                if u not in u_list: u_list.append(u)
                if i not in i_list: i_list.append(i)
                err = err + (dataSet[u][i] - np.dot(P[u],Q[i])) ** 2
    reg1 = sum([np.dot(P[u],P[u]) for u in u_list])
    reg2 = sum([np.dot(Q[i],Q[i]) for i in i_list])
    totalvalue = err + lamda * (reg1 + reg2)
    return totalvalue
def function_reg(dataSet,P,Q):
    totalvalue = 0
    for u in range(len(dataSet)):
        for i in range(len(dataSet[u])):
            if dataSet[u][i] > 0:
                err = (dataSet[u][i] - np.dot(P[u], Q[i])) ** 2 + lamda * (np.dot(P[u],P[u]) + np.dot(Q[i],Q[i]))
                totalvalue = totalvalue + err
    return totalvalue
def ALS_vector(target,Mat,type='P'):
    inx = []
    r = []
    if type == 'P':
        inx = [i for i,x in enumerate(TrainSetRatings[target]) if x > 0]
        r = TrainSetRatings[target][inx]
    elif type == 'Q':
        inx = [i for i, x in enumerate(TrainSetRatings[:, target]) if x > 0]
        r = TrainSetRatings[:,target][inx]
    if not inx:
        vector = np.zeros(n)
    else:
        NewMat = Mat[inx]
        MTM = np.dot(NewMat.T,NewMat)
        lamdaI = np.eye(MTM.shape[0]) * len(inx) * lamda
        vector = np.linalg.solve((MTM + lamdaI), np.dot(NewMat.T,r.T))
    return vector
#ALS
def ALS(TrainSetRatings,TestSetRatings,P,Q,n):
    testError = Error(TestSetRatings, P, Q, n)
    trainError = Error(TrainSetRatings, P, Q, n)
    # prefvalue = function_reg(TrainSetRatings, P, Q)
    pretestMAE_Error, pretestRMSE_Error = testError
    pretest_PPR = rating_ability(TestSetRatings, P, Q, n)
    pretrainMAE_Error, pretrainRMSE_Error = trainError
    print('pre_train_mae : {}'.format(pretrainMAE_Error))
    print('pre_train_rmse : {}'.format(pretrainRMSE_Error))
    print('-------------------------------------------')
    print('pre_test_mae : {}'.format(pretestMAE_Error))
    print('pre_test_rmse : {}'.format(pretestRMSE_Error))
    print('pre_test_PPR : {}'.format(pretest_PPR))
    print('-------------------------------------------')
    # print('pre_favlue:{}'.format(prefvalue))
    MAE_testlist = [pretestRMSE_Error]
    RMSE_testlist = [pretestMAE_Error]
    PPR_list = [pretest_PPR]
    MAE_trainlist = [pretrainMAE_Error]
    RMSE_trainlist = [pretrainRMSE_Error]
    # fvaluelist = [prefvalue]
    itera = 0
    for iter in range(1,iter_array[-1]+1):
        # OPTION1:update matrix
        # #update P
        # partP1 = np.dot(TrainSetRatings,Q)
        # partP2 = np.dot(Q.T,Q) + lamda * np.identity(n)
        # partP2 = np.linalg.inv(partP2)
        # P = np.dot(partP1,partP2)
        # #update Q
        # partQ1 = np.dot(TrainSetRatings.T,P)
        # partQ2 = np.dot(P.T,P) + lamda * np.identity(n)
        # partQ2 = np.linalg.inv(partQ2)
        # Q = np.dot(partQ1,partQ2)
        # OPTION2:update vector
        for u in range(len(TrainSetRatings)):
            P[u,:] = ALS_vector(u,Q,'P')
        for i in range(len(TrainSetRatings[0])):
            Q[i,:] = ALS_vector(i,P,'Q')
        # OPTION3:update element 
        #update P
        # for u in range(len(P)):
        #     for k in range(len(P[u])):
        #         fenzi = 0
        #         part2 = 0
        #         count = 0
        #         for i in range(len(TrainSetRatings[u])):
        #             if TrainSetRatings[u][i] > 0:
        #                 fenzi = fenzi + Q[i][k] * (TrainSetRatings[u][i] - (np.dot(P[u],Q[i]) - P[u][k] * Q[i][k]))
        #                 part2 = part2 + Q[i][k] ** 2
        #                 count = count + 1
        #         fenmu = part2 + lamda * count
        #         if not fenmu : P[u][k] = 0
        #         else:
        #             P[u][k] = fenzi/fenmu
        # # print(function_reg(TrainSetRatings,P,Q))
        #update Q
        # for i in range(len(Q)):
        #     for k in range(len(Q[i])):
        #         fenzi = 0
        #         part2 = 0
        #         count = 0
        #         for u in range(len(TrainSetRatings[:,i])):
        #             if TrainSetRatings[u][i] > 0:
        #                 fenzi = fenzi + P[u][k] * (TrainSetRatings[u][i] - (np.dot(P[u],Q[i]) - P[u][k] * Q[i][k]))
        #                 part2 = part2 + P[u][k] ** 2
        #                 count = count + 1
        #         fenmu = part2 + lamda * count
        #         if not fenmu: Q[i][k] = 0
        #         else:
        #             Q[i][k] = fenzi/fenmu
        # print((function_reg(TrainSetRatings,P,Q)))
        #Error
        MAE_test, RMSE_test = Error(TestSetRatings, P, Q, n)
        PPR = rating_ability(TestSetRatings, P, Q, n)
        MAE_train, RMSE_train = Error(TrainSetRatings, P, Q, n)
        # fvalue = function_reg(TrainSetRatings, P, Q)
        MAE_testlist.append(MAE_test)
        RMSE_testlist.append(RMSE_test)
        PPR_list.append(PPR)
        MAE_trainlist.append(MAE_train)
        RMSE_trainlist.append(RMSE_train)
        # fvaluelist.append(fvalue)
        itera = itera + 1
        if itera in iter_array:
            print('iteration:{}'.format(itera))
            print('train mae:{}'.format(MAE_train))
            print('train rmse:{}'.format(RMSE_train))
            print('-------------------------------------------')
            print('test mae:{}'.format(MAE_test))
            print('test rmse:{}'.format(RMSE_test))
            print('test_PPR:{}'.format(PPR))
            print('-------------------------------------------')
            # print('favlue:{}'.format(fvalue))
    print("Completely")
    return MAE_testlist, RMSE_testlist
#Learning curve
# def print_curve(propose_rmse,compare_rmse,iter_array):
#     plt.plot(iter_array, compare_rmse,label='train', linewidth=5)
#     plt.plot(iter_array, propose_rmse,label='test', linewidth=5)
#     plt.title('ml;n=10;reg=0.1')
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.yticks(np.arange(0.7,3,0.2),fontsize=16)
#     plt.xlabel('iterations', fontsize=10)
#     plt.ylabel('RMSE', fontsize=10)
#     plt.legend(loc='best', fontsize=20)
#     plt.show()
# def save_rmse(rmse,path):
#     try:
#         f_name = open(path,'wb')
#         pickle.dump(rmse,f_name)
#     finally:
#         f_name.close()
#Main function
if __name__ == '__main__':
    time_start = time.time()
    np.random.seed(0)
    trainsetFile = r"./trainset_100k.csv"
    testsetFile = r"./testset_100k.csv"
    TrainSetRatings = readTrainSet(trainsetFile)
    TestSetRatings = readTestSet(testsetFile,TrainSetRatings)
    #Latent features
    n = 10
    # Regularization parameter
    lamda = 0.1
    P,Q = initPQ(TrainSetRatings,n)
    iter_array = np.arange(20)
    test_MAE,test_RMSE = ALS(TrainSetRatings,TestSetRatings,P,Q,n)
    time_end = time.time()
    print("time:", time_end - time_start)








