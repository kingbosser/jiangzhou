#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt
import copy
import math
import pickle
import warnings
import time
#Trainset 
def readTrainSet(trainsetFile):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_table(trainsetFile, sep=',',names=names)
    user_list = list(df.user_id.unique())
    user_list.sort(reverse=True)
    user_maxnum = user_list[0]
    item_list = list(df.item_id.unique())
    item_list.sort(reverse=True)
    item_maxnum = item_list[0]
    TrainSetRatings = np.zeros((user_maxnum+1, item_maxnum+1))
    for row in df.itertuples():
        TrainSetRatings[row[1]][row[2]] = row[3]
    return TrainSetRatings
#Testset
def readTestSet(testsetFile,TrainSetRatings):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_table(testsetFile, sep=',', names=names)
    rows = len(TrainSetRatings)
    columns = len(TrainSetRatings[0])
    TestSetRatings = np.zeros((rows,columns))
    for row in df.itertuples():
        TestSetRatings[row[1]][row[2]] = row[3]
    return TestSetRatings
#User average
def get_usermean(TrainSetRatings):
    return np.array([np.mean(TrainSetRatings[u,:][TrainSetRatings[u,:]!= 0])\
                          for u in range(len(TrainSetRatings))]).reshape(-1,1)
#Item average
def get_itemmean(TrainSetRatings):
    return np.array([np.mean(TrainSetRatings[:,i][TrainSetRatings[:,i]!= 0])\
                          for i in range(len(TrainSetRatings[0]))]).reshape(-1,1)
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
def Error(Ratings,P,Q):
    totalError = 0
    count = 0
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:continue
            count = count + 1
            # hat_rui = np.dot(P[u],Q[i]) + usermean[u]
            hat_rui = np.dot(P[u], Q[i]) + itemmean[i]
            err = (Ratings[u][i] - hat_rui) ** 2
            totalError = totalError + err
            if math.isnan(totalError):
                print('error')
    RMSE = (totalError / count) ** 0.5
    return RMSE
#Objective function：
def function_reg(dataSet,P,Q):
    totalvalue = 0
    for u in range(len(dataSet)):
        for i in range(len(dataSet[u])):
            if dataSet[u][i] > 0:
                # hat_rui = np.dot(P[u], Q[i]) + usermean[u]
                hat_rui = np.dot(P[u], Q[i]) + itemmean[i]
                err = (dataSet[u][i] - hat_rui) ** 2 + lamda * (np.dot(P[u],P[u]) + np.dot(Q[i],Q[i]))
                totalvalue = totalvalue + err
    return totalvalue
#SGD
def SGD(R,P,Q):
    itera = 0
    step = 0.01
    curError = Error(TestSetRatings, P, Q)
    trainError = Error(TrainSetRatings, P, Q)
    prefvalue = function_reg(TrainSetRatings, P, Q)
    pretestError = curError
    pretrainError = trainError
    print('pre_train_rmse : {}'.format(pretrainError))
    print('pre_test_rmse : {}'.format(pretestError))
    print('pre_favlue:{}'.format(prefvalue))
    curErrorlist = [pretestError]
    trainErrorlist = [pretrainError]
    fvaluelist = [prefvalue]
    nonzero_row, nonzero_column = R.nonzero()
    nonzero_num = len(nonzero_row)
    sample_arr = np.arange(nonzero_num)
    for iter in range(1,iter_array[-1]+1):
        np.random.shuffle(sample_arr)
        for idx in sample_arr:
            u = nonzero_row[idx]
            i = nonzero_column[idx]
            # rui = R[u][i] - usermean[u]
            rui = R[u][i] - itemmean[i]
            eui = rui - np.dot(P[u],Q[i])
            gradPu = 2 * (lamda * P[u] - eui * Q[i])
            gradQi = 2 * (lamda * Q[i] - eui * P[u])
            P[u] = P[u] - step * gradPu
            Q[i] = Q[i] - step * gradQi
        curError = Error(TestSetRatings, P, Q)
        trainError = Error(TrainSetRatings, P, Q)
        fvalue = function_reg(TrainSetRatings, P, Q)
        curErrorlist.append(curError)
        trainErrorlist.append(trainError)
        fvaluelist.append(fvalue)
        itera = itera + 1
        if itera in iter_array:
            print('iteration:{}'.format(itera))
            print('train rmse:{}'.format(trainError))
            print('test rmse:{}'.format(curError))
            print('favlue:{}'.format(fvalue))
    print("Completely")
    return P, Q, curErrorlist, trainErrorlist
#Neural network model
def ANN(P,Q,Neurons):
    user_idx, item_idx = TrainSetRatings.nonzero()
    # mean_array = usermean[user_idx].ravel()
    mean_array = itemmean[item_idx].ravel()
    X = np.concatenate((P[user_idx],Q[item_idx]),axis=1)
    y = TrainSetRatings[TrainSetRatings != 0] - mean_array
    clf = MLPRegressor(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=Neurons, random_state=1)
    model = clf.fit(X,y)
    return model
#NB
def NB(P,Q):
    user_idx, item_idx = TrainSetRatings.nonzero()
    # mean_array = usermean[user_idx].ravel()
    mean_array = itemmean[item_idx].ravel()
    X = np.concatenate((P[user_idx], Q[item_idx]), axis=1)
    y = np.rint(TrainSetRatings[TrainSetRatings != 0] - mean_array)
    clf = GaussianNB()
    model = clf.fit(X, y)
    return model
#RandomForest
def RF(P,Q,Trees):
    user_idx, item_idx = TrainSetRatings.nonzero()
    # mean_array = usermean[user_idx].ravel()
    mean_array = itemmean[item_idx].ravel()
    X = np.concatenate((P[user_idx], Q[item_idx]), axis=1)
    y = TrainSetRatings[TrainSetRatings != 0] - mean_array
    clf = RandomForestRegressor(n_estimators=Trees,random_state=1)
    model = clf.fit(X, y)
    return model
#MAE和RMSE
def get_MAE_RMSE(Ratings,P,Q,model):
    maeError = []
    rmseError = []
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:
                continue
            xpred = np.concatenate((P[u],Q[i])).reshape(1,-1)
            # hat_rui = model.predict(xpred) + usermean[u]
            hat_rui = model.predict(xpred) + itemmean[i]
            hat_rui = np.clip(hat_rui, 1., 5.)
            error = math.fabs(Ratings[u][i] - hat_rui)
            maeError.append(error)
            rmseError.append(error ** 2)
    MAE = sum(maeError) / len(maeError)
    RMSE = math.sqrt(sum(rmseError) / len(rmseError))
    return MAE, RMSE
#Accuracy
def getF_value(Ratings,P,Q,model):
    actual_num = 0.0
    perfect_num = 0.0
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:
                continue
            else:
                actual_num += 1.0
                xpred = np.concatenate((P[u], Q[i])).reshape(1, -1)
                # hat_rui = model.predict(xpred) + usermean[u]
                hat_rui = model.predict(xpred) + itemmean[i]
                hat_rui = float(np.clip(hat_rui, 1., 5.))
                if (round(Ratings[u][i]) == round(hat_rui)):
                    perfect_num += 1.0
    SPR = 1
    PPR = perfect_num / actual_num
    return SPR, PPR
#Main function
if __name__ == '__main__':
    time_start = time.time()
    np.random.seed(0)
    warnings.filterwarnings("ignore")
    trainsetFile = r"./trainset_100k.csv"
    testsetFile = r"./testset_100k.csv"
    TrainSetRatings = readTrainSet(trainsetFile)
    TestSetRatings = readTestSet(testsetFile,TrainSetRatings)
    #Latent feature
    n = 8
    #Regularization parameter
    lamda = 0.1
    P, Q = initPQ(TrainSetRatings,n)
    iter_array = np.arange(1,20)
    #option1:user average
    # usermean = get_usermean(TrainSetRatings)
    # usermean[np.isnan(usermean)] = 0
    #option2:item average
    itemmean = get_itemmean(TrainSetRatings)
    itemmean[np.isnan(itemmean)] = 0


    P, Q, test_RMSE,train_RMSE = SGD(TrainSetRatings,P,Q)

    trees = 50
    model = RF(P,Q,trees)
    error_rf = get_MAE_RMSE(TestSetRatings, P, Q, model)
    precision_rf = getF_value(TestSetRatings, P, Q, model)
    print('rf :\nMAE: {} RMSE : {}'.format(*error_rf))
    print('SPR: {} PPR : {}'.format(*precision_rf))
    time_end = time.time()
    print("time:", time_end - time_start)

