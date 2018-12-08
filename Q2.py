#Copyright: Copyright (c) 2018
#Created on 14/11/2018
#Author:Yihang OU
#Version 1.0
#Title: Question 2 for CW1 for Supervised Learning 18/19 in UCL

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

#Question2(a)
def Question2A():
    #y1=sin(2*pi*x)^2+noise
    np.random.seed(50)
    noise=np.random.normal(0,0.07,30)
    x1=np.random.uniform(0,1,30)
    y1=np.square(np.sin(2*math.pi*x1))
    i=0
    while(i<y1.size):
        y1[i]=y1[i]+noise[i]
        i=i+1

    #y_t=sin(2*pi*x)^2
    xn=np.linspace(0,1,100)
    y_t=np.square(np.sin(2*math.pi*xn))
    #weight w when k=2,k=5,k=10,k=14,k=18
    w1=fit(x1,y1,1)
    w2=fit(x1,y1,4)
    w3=fit(x1,y1,9)
    w4=fit(x1,y1,13)
    w5=fit(x1,y1,17)
    plt.figure(1)
    plt.scatter(x1,y1)
    xn=np.linspace(0,1,100)
    Y_t1=predict(xn,w1,1)
    Y_t2=predict(xn,w2,4)
    Y_t3=predict(xn,w3,9)
    Y_t4=predict(xn,w4,13)
    Y_t5=predict(xn,w5,17)

    l1,=plt.plot(xn,y_t)
    l2,=plt.plot(xn,Y_t1)
    l3,=plt.plot(xn,Y_t2)
    l4,=plt.plot(xn,Y_t3)
    l5,=plt.plot(xn,Y_t4)
    l6,=plt.plot(xn,Y_t5)
    plt.xlabel('X')
    plt.ylabel("Y")
    plt.title("Question2a")
    plt.legend(handles=[l1, l2, l3, l4,l5,l6],
               labels=['sin(2*pi*x)^2', 'k=2', 'k=5',
                       'k=10','k=14','k=18'], loc='best')
    plt.xlim(0,1.1)
    plt.ylim(-1.5,2.5)
    plt.show()

#Question2B
def Question2B():
    np.random.seed(20)
    #y1=sin(2*pi*x)^2+noise
    noise=np.random.normal(0,0.07,30)
    x1=np.random.uniform(0,1,30)
    y1=np.square(np.sin(2*math.pi*x1))

    i=0
    while(i<y1.size):
        y1[i]=y1[i]+noise[i]
        i=i+1
    #create a mse for MSE
    mse=[]
    for i in range(18):
        w=fit(x1,y1,i)
        Y_t=predict(x1,w,i)
        mse.append(math.log(cal_mse(y1,Y_t)))

    plt.figure(1)
    plt.title("Question2b")
    plt.xlabel("k")
    plt.ylabel("log(MSE)")
    x_mse=np.linspace(1,18,18)
    plt.plot(x_mse,mse)
    plt.show()

#Question2C
def Question2C():
    #when k=1,...,18
    #train data generate the w
    def train(k):
        np.random.seed(30)
        s = np.random.normal(0, 0.07, 30)

        x1 = np.random.uniform(0, 1, 30)
        x1.sort()
        y1 = np.square(np.sin(2 * math.pi * x1))

        i = 0
        while (i < y1.size):
            y1[i] = y1[i] + s[i]
            i = i + 1

        w = fit(x1, y1, k)
        return w

    #use the w from train to compute the y_predict
    #compute the MSE in different k
    def test(k):
        w = train(k)
        np.random.seed(30)
        s = np.random.normal(0, 0.07, 1000)

        x1 = np.random.uniform(0, 1, 1000)
        y1 = np.square(np.sin(2 * math.pi * x1))
        i = 0
        while (i < y1.size):
            y1[i] = y1[i] + s[i]
            i = i + 1

        y_t = predict(x1, w, k)
        y_t = np.array(y_t)
        mse = cal_mse(y_t, y1)
        mse_log = math.log(mse)
        return mse_log

    #create list for mse
    res_mse = []
    for i in range(18):
        res_mse.append(test(i))

    plt.figure(1)
    x = np.linspace(1, 18, 18)
    plt.plot(x, res_mse)
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.title("Question2C")
    plt.show()

def Question2D():
    #create two list for MSE_train and MSE_test
    MSE_train=[]
    MSE_test=[]
    #loop for k from 1 to 18
    for k in range(18):
        print("current K is "+str(k) )
        mse_train=0
        mse_test=0
        #run train and test 100 times
        #compute each time mse
        for m in range(100):
            #train data
            s = np.random.normal(0, 0.07, 30)

            x1 = np.random.uniform(0, 1, 30)
            y1 = np.square(np.sin(2 * math.pi * x1))
            i = 0
            while (i < y1.size):
                y1[i] = y1[i] + s[i]
                i = i + 1

            w=fit(x1,y1,k)
            Y_t1=predict(x1,w,k)
            #train finish!
            #compute mse of train
            mse_train+=cal_mse(y1,Y_t1)

            #test data
            s2=np.random.normal(0,0.07,1000)
            x2=np.random.uniform(0,1,1000)
            y2=np.square(np.sin(2*math.pi*x2))
            n=0
            while(n<y2.size):
                y2[i]=y2[i]+s2[i]
                n=n+1
            #"start test!"
            Y_t=predict(x2,w,k)
            mse_test=mse_test+cal_mse(y2,Y_t)

        #compute the log(average of MSE)
        Mse_train=math.log(mse_train/100)
        Mse_test=math.log(mse_test/100)
        MSE_test.append(Mse_test)
        MSE_train.append(Mse_train)
    plt.figure(1)
    x=np.linspace(1,18,18)
    l1,=plt.plot(x,np.array(MSE_train))
    l2,=plt.plot(x,np.array(MSE_test))
    plt.title("Question2D")
    plt.xlabel("k")
    plt.ylabel("log(avg of MSE)")
    plt.legend(handles=[l1, l2,], labels=['MSE_train', 'MSE_test'], loc='best')
    plt.show()

if __name__ == '__main__':
    #Question2A()
    #Question2B()
    #Question2C()
    #Question2D()
