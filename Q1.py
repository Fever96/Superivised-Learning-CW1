#Copyright: Copyright (c) 2018
#Created on 14/11/2018
#Author:Yihang OU
#Version 1.0
#Title: Question 1 for CW1 for Supervised Learning 18/19 in UCL

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#generate the feature map
#use input data x and the dimension of polynomial bases k
def feature_map(x,k):
    X=[]
    for i in range(k+1):
       X.append(x**i)
    X=np.mat(X).T
    X=X.astype(np.float32)
    return X

#generate the weight w
#use the feature map X=feature_map(x,k) and the goal number y
def fit(x,y,k):
    X=np.around(feature_map(x,k).astype('float64'),decimals = 7) #feature map
    Y = np.array(y).reshape((len(y), 1))
    XT = X.transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), Y)
    return w

#predict function
#use the feature x and weight w and the dimension of polynomial bases k
def predict(x,w,k):
    X=feature_map(x,k)
    #print(X)
    y=X@w
    return y

#calculate MSE
#MSE=SSE/n
#input is y_true and y_predict
def cal_mse(y_t,y_p):
    result=[]
    if(y_t.size!=y_p.size):
        print("Input error")
    for i in range(len(y_t)):
        mse1=pow((y_t[i]-y_p[i]),2)
        result.append(mse1)
    mse=sum(result)/len(y_t)
    return mse

#Question 1
def Q1():
    #data
    data = np.array([(1, 3), (2, 2), (3, 0), (4, 5)])
    x = data[:, 0]
    y = data[:, 1]

    #wn is the weight of linear regression
    w1 = fit(x, y, 0)   #k=1
    w2 = fit(x, y, 1)   #k=2
    w3=fit(x, y, 2)     #k=3
    w4=fit(x, y, 3)     #k=4

    #generate the equation of different linear function
    print("k=1: ("+str(w1[0])+")")
    print("k=2: ("+str(w2[1])+")*x"+"+("+str(w2[0])+")")
    print("k=3: ("+str(w3[2])+")*x^2+("+str(w3[1])+")*x+("+str(w3[0])+")")
    print("k=4: ("+str(w4[3])+")*x^3+("+str(w4[2])+")*x^2+("+str(w4[1])+")*x+("+str(w4[0])+")")

    #plot a picture to illustrate the linear function
    xn=np.linspace(0,4.5,100)
    Y_t1=predict(xn,w1,0)
    Y_t2=predict(xn,w2,1)
    Y_t3=predict(xn,w3,2)
    Y_t4=predict(xn,w4,3)
    plt.figure(1)
    plt.scatter(x,y)
    l1,=plt.plot(xn,Y_t1)
    l2,=plt.plot(xn,Y_t2)
    l3,=plt.plot(xn,Y_t3)
    l4,=plt.plot(xn,Y_t4)
    plt.legend(handles=[l1, l2,l3,l4 ], labels=['k=1', 'k=2','k=3','k=4'], loc='best')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Question1")
    #predict Y with different weight w and different polynomial bases
    Y_t1=predict(x,w1,0)
    Y_t2=predict(x,w2,1)
    Y_t3=predict(x,w3,2)
    Y_t4=predict(x,w4,3)

    print("k=1,the mse="+str(cal_mse(y,Y_t1)))
    print("k=2,the mse="+str(cal_mse(y,Y_t2)))
    print("k=3,the mse="+str(cal_mse(y,Y_t3)))
    print("k=4,the mse="+str(cal_mse(y,Y_t4)))
    plt.show()

if __name__ == '__main__':
    Q1()