#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn import svm
import copy
# import matplotlib.pyplot as plt
import time

#Trainset
def readTrainSet(trainsetFile):
    names = ['user_id', 'item_id', 'rating']
    data = pd.read_table(trainsetFile, sep=',', names=names)
    user_maxnum, item_maxnum = int(data['user_id'].max()), int(data['item_id'].max())
    popular_items = np.array(pd.value_counts(data['item_id'],sort=True).index)
    TrairSetRatings = np.zeros((user_maxnum, item_maxnum))
    popular_items = popular_items - 1
    for row in data.itertuples():
        TrairSetRatings[row[1]-1][row[2]-1] = row[3]
    return TrairSetRatings , popular_items

#Testset
def readTestSet(testsetFile,TrainSetRatings):
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_table(testsetFile, sep=',', names=names)
    rows = len(TrainSetRatings)
    columns = len(TrainSetRatings[0])
    TestSetRatings = np.zeros((rows,columns))
    for row in df.itertuples():
        TestSetRatings[row[1]-1][row[2]-1] = row[3]
    return TestSetRatings

#Cosine similarity between items
def cosine_similarity(v,p):
    co_user = np.array([ _ for _ in TrainSetRatings[:, v].nonzero()[0] \
                        if _ in TrainSetRatings[:, p].nonzero()[0]])
    if len(co_user) == 0:
        cos_vp = 0
    else:
        vector_v = TrainSetRatings[:,v][co_user]
        vector_p = TrainSetRatings[:,p][co_user]
        sumtop = np.dot(vector_v,vector_p)
        sumbot = np.linalg.norm(vector_v) * np.linalg.norm(vector_p)
        cos_vp = sumtop / sumbot
    return cos_vp

#Similarity matrix between item and landmarks
def similarity_matrix():
    S = np.zeros((len(TrainSetRatings[0]),len(landmarks)))
    for i in range(len(TrainSetRatings[0])):
        for j in range(len(landmarks)):
            v = i
            p = landmarks[j]
            S[i][j] = cosine_similarity(v,p)
        if (i % 300 == 0 and i != 0):
            print('Already completed %d items'%i)
    # Normalizaion
    for col in range(len(S[0])):
        sortlist = sorted(S[:,col])
        a = sortlist[0]
        b = sortlist[-1]
        for item in range(len(S[:,col])):
            if a == b:
                S[item][col] = 0
            else:
                S[item][col] = (S[item][col] - a) / (b-a)
    return S

#SVR
def learning_svr_model(SimMat,Gamma=10):
    print(20*'*'+'Learning model'+20*'*')
    UserModel = {}
    for u in range(len(TrainSetRatings)):
        y = TrainSetRatings[u][TrainSetRatings[u] != 0]
        X = SimMat[TrainSetRatings[u].nonzero()[0]]
        svr_rbf = svm.SVR(kernel='rbf', gamma=Gamma)
        model = svr_rbf.fit(X,y)
        UserModel['%d'%u] = model
        if (u % 300 == 0 and u != 0):
            print('Already completed %d users'%u)
    return UserModel

#MAE and RMSE
def Error(Ratings,SimMat,UserModel):
    print(20 * '*' + 'error' + 20 * '*')
    maeError = []
    rmseError = []
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:
                continue
            hat_rui = UserModel[str(u)].predict(SimMat[i, :].reshape(1, -1))
            hat_rui = np.clip(hat_rui, 1., 5.)
            error = math.fabs(Ratings[u][i] - hat_rui)
            maeError.append(error)
            rmseError.append(error ** 2)
    MAE = sum(maeError) / len(maeError)
    RMSE = math.sqrt(sum(rmseError) / len(rmseError))
    return MAE, RMSE

#Accuracy
def getF_value(Ratings,SimMat,UserModel):
    actual_num = 0.0
    perfect_num = 0.0
    for u in range(len(Ratings)):
        for i in range(len(Ratings[u])):
            if Ratings[u][i] == 0:
                continue
            else:
                actual_num += 1.0
                hat_rui = UserModel[str(u)].predict(SimMat[i, :].reshape(1, -1))
                hat_rui = float(np.clip(hat_rui, 1., 5.))
                if (round(Ratings[u][i]) == round(hat_rui)):
                    perfect_num += 1.0
    SPR = 1
    PPR = perfect_num / actual_num
    return SPR, PPR

#Main function
if __name__ == '__main__':
    time_start = time.time()
    trainsetFile = r"./trainset_100k.csv"
    testsetFile = r"./testset_100k.csv"
    TrainSetRatings , popular_items = readTrainSet(trainsetFile)
    TestSetRatings = readTestSet(testsetFile,TrainSetRatings)
    # Landmarks
    L = 20
    landmarks = popular_items[:L][::-1]
    # Similarity matrix
    simmat = similarity_matrix()
    # SVR
    Gama = 10
    usermodel = learning_svr_model(simmat,Gama)
    # Evaluation
    MAE, RMSE = Error(TestSetRatings, simmat, usermodel)
    SPR, PPR = getF_value(TestSetRatings, simmat, usermodel)
    print('MAE : %.4f RMSE : %.4f'%(MAE,RMSE))
    print('SPR : %.4f PPR : %.4f'%(SPR,PPR))
    time_end = time.time()
    print("time:", time_end - time_start)