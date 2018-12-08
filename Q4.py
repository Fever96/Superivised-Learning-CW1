#Copyright: Copyright (c) 2018
#Created on 14/11/2018
#Author:Yihang OU
#Version 1.0
#Title: Question 4 for CW1 for Supervised Learning 18/19 in UCL

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#load the data
input=sio.loadmat('boston.mat')
data=input['boston']

#generate the feature map
#use input data x and the dimension of polynomial bases k
def feature_map(x,k):
    X=[]
    for i in range(k+1):
       X.append(x**i)
    X=np.mat(X).T
    X=X.astype(np.float32)
    return X

#use for linear regression for many attributes
def fit2(x,y):
    X=x
    Y=y.reshape((len(y),1))
    XT=x.transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), Y)
    return w

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

#predict function in many attributes
def predict2(x,w):
    y=np.dot(x,w)
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

#Calculte MSE for train set and test set at the same time
def Calculate_MSE(y_train,y_test,y_mean):
    result_train=[]
    result_test=[]
    for i in range(len(y_train)):
        mse1=pow((y_train[i]-y_mean),2)
        result_train.append(mse1)
    train_mse=sum(result_train)/len(y_train)
    for i in range(len(y_test)):
        mse2=pow((y_test[i]-y_mean),2)
        result_test.append(mse2)

    test_mse=sum(result_test)/len(y_test)
    return train_mse,test_mse

#for Question A
def QuestionA():
    MSE=[0,0]
    for loop in range(20):
        #split data
        number1=round(data.shape[0]*2/3)
        np.random.shuffle(data)
        training, test = data[:number1,:], data[number1:,:]

        y_train=training[:,13]
        y_test=test[:,13]
        y_mean=y_train.mean()
        mse_train,mse_test=Calculate_MSE(y_train,y_test,y_mean)
        MSE[0]+=mse_train
        MSE[1]+=mse_test
    MSE[0]=MSE[0]/20
    MSE[1]=MSE[1]/20

    print("MSE of training set is " +str(MSE[0]))
    print("MSE of test set is "+str(MSE[1]))

#QuestionC
def QuestionC():
    mse=[0.0]*13
    mse=np.array(mse)
    for i in range(len(mse)):
        print("feature column now "+str(i))
        for loop in range(20):
            #split data
            number1=round(data.shape[0]*2/3)
            np.random.shuffle(data)
            training, test = data[:number1,:], data[number1:,:]
            y_train=training[:,13]
            x_train=training[:,i]
            y_test=test[:,13]
            x_test=test[:,0]
            #train
            w=fit(x_train,y_train,1)
            #predict and test
            y_predict1=predict(x_test,w,1)
            mse1=cal_mse(y_t=y_test,y_p=y_predict1)
            mse[i]+=mse1
        mse[i]=mse[i]/20
    #figure
    plt.figure(1)
    x=np.linspace(1,13,13)
    plt.xlabel("the nth attributes")
    plt.ylabel("MSE")
    plt.title("Question4C")
    plt.show()

#QuestionD
def QuestionD():
    mse=0
    for loop in range(20):
        print(loop)
        number1 = round(data.shape[0] * 2 / 3)
        number2 = data.shape[0] - number1
        np.random.shuffle(data)
        training, test = data[:number1, :], data[number1:, :]
        y_train = training[:, 13]
        x_train = training[:, :13]
        x_test=test[:,:13]
        y_test=test[:,13]
        ones1=np.ones((number1,1))
        ones2=np.ones((number2,1))
        x_train=np.hstack((x_train,ones1))
        x_test=np.hstack((x_test,ones2))
        w = fit2(x_train,y_train)
        y=predict2(x_test,w)
        mse1=cal_mse(y_test,y)
        mse+=mse1
    print("mse is "+str(mse/20))

#Question for 5D
#repeate exercise 4a over 20 random (2/3,1/3) splits of your data
def Question5D_1():
    MSE_test=[0.0]*20
    MSE_train=[0.0]*20
    MSE_train=np.array(MSE_train)
    MSE_test=np.array(MSE_test)
    for loop in range(20):
        #split data
        number1=round(data.shape[0]*2/3)
        np.random.shuffle(data)
        training, test = data[:number1,:], data[number1:,:]
        y_train=training[:,13]
        y_test=test[:,13]
        y_mean=y_train.mean()
        #calculte the MSE
        mse_train,mse_test=Calculate_MSE(y_train,y_test,y_mean)
        MSE_train[loop]=mse_train
        MSE_test[loop]=mse_test

    Mse_train=MSE_train.mean()
    Mse_test=MSE_test.mean()
    STD_train=np.std(MSE_train,ddof=1)
    STD_test=np.std(MSE_test,ddof=1)

    #print the results
    print("The average MSE of training set is " +str(Mse_test))
    print("The average MSE of test set is "+str(Mse_train))
    print("The standard deviarions of train error is " + str(STD_train))
    print("The standard deviarions of test error is " + str(STD_test))

#Question for 5D
#repeate exercise 4c over 20 random (2/3,1/3) splits of your data
def Question5D_2():
    result=[[0.0]*4]*13
    result=np.array(result)
    for i in range(13): #i is the nth attribute
        print("feature column now "+str(i))
        MSE_test = [0.0] * 20
        MSE_train = [0.0] * 20
        MSE_train = np.array(MSE_train)
        MSE_test = np.array(MSE_test)
        for loop in range(20):
            #split data
            number1=round(data.shape[0]*2/3)
            np.random.shuffle(data)
            training, test = data[:number1,:], data[number1:,:]
            y_train=training[:,13]
            x_train=training[:,i]
            y_test=test[:,13]
            x_test=test[:,0]
            #train
            w=fit(x_train,y_train,1)
            #predict and test
            y_predict1=predict(x_test,w,1)
            y_predict2=predict(x_train,w,1)
            #calculate MSE
            mse_test=cal_mse(y_t=y_test,y_p=y_predict1)
            mse_train=cal_mse(y_train,y_predict2)
            MSE_test[i] = mse_test
            MSE_train[i] = mse_train

        Mse_train = MSE_train.mean()
        Mse_test = MSE_test.mean()
        STD_train = np.std(MSE_train, ddof=1)
        STD_test = np.std(MSE_test, ddof=1)

        result[i,0]=Mse_train
        result[i,1]=Mse_test
        result[i,2]=STD_train
        result[i,3]=STD_test

    #print the results
    for m in range(13):
        print("With Attribute "+str(m+1))
        print("The average MSE of training set is " + str(result[m,0]))
        print("The average MSE of test set is " + str(result[m,1]))
        print("The standard deviarions of train error is " + str(result[m,2]))
        print("The standard deviarions of test error is " + str(result[m,3]))

#Question for 5D
#repeate exercise 4d over 20 random (2/3,1/3) splits of your data
def Question5D_3():
    #create two list for store the MSE_train and MSE_test
    MSE_test = [0.0] * 20
    MSE_train = [0.0] * 20
    MSE_train = np.array(MSE_train)
    MSE_test = np.array(MSE_test)
    for loop in range(20):
        #print(loop)
        #split data
        number1 = round(data.shape[0] * 2 / 3)
        number2 = data.shape[0] - number1
        np.random.shuffle(data)     #shuffle data so that each time data set is different
        training, test = data[:number1, :], data[number1:, :]
        y_train = training[:, 13]
        x_train = training[:, :13]
        x_test=test[:,:13]
        y_test=test[:,13]
        ones1=np.ones((number1,1))
        ones2=np.ones((number2,1))
        x_train=np.hstack((x_train,ones1))
        x_test=np.hstack((x_test,ones2))
        #train
        w = fit2(x_train,y_train)
        #predict &test
        y1=predict2(x_test,w)
        y2=predict2(x_train,w)
        #calculate the MSE
        mse_test=cal_mse(y_test,y1)
        mse_train=cal_mse(y_train,y2)
        MSE_test[loop] = mse_test
        MSE_train[loop] = mse_train

    Mse_train = MSE_train.mean()
    Mse_test = MSE_test.mean()
    STD_train = np.std(MSE_train, ddof=1)
    STD_test = np.std(MSE_test, ddof=1)

    print("The average MSE of training set is " +str(Mse_test))
    print("The average MSE of test set is "+str(Mse_train))
    print("The standard deviarions of train error is " + str(STD_train))
    print("The standard deviarions of test error is " + str(STD_test))

if __name__ == '__main__':
    #QuestionA()
    #QuestionC()
    #QuestionD()
    #Question5D_1()
    #Question5D_2()
    #Question5D_3()