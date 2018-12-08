#Copyright: Copyright (c) 2018
#Created on 14/11/2018
#Author:Yihang OU
#Version 1.0
#Title: Question 5 for CW1 for Supervised Learning 18/19 in UCL

import scipy.io as sio
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#input data
input=sio.loadmat('boston.mat')
data=input['boston']

#compute the distance between two vectors
def distance(x1,x2):
    distance=np.linalg.norm(x1-x2)
    return distance

#compute the kernel function
#return kernel function k
def kernel(x1,x2,sigma):
    m1=x1.shape[0]
    m2=x2.shape[0]
    k=[[0.0]*m2]*m1
    k=np.array(k)
    for i in range(m1):
        for j in range(m2):
            k[i][j] = np.exp(-pow(distance(x1[i], x2[j]), 2) / (2 * pow(sigma, 2)))
    return k

#compute a_star
def a_star(x,y,K,gamma):
    m,n=x.shape
    I=np.eye(m,dtype=float)
    temp=gamma*m*I
    a=np.linalg.inv(K+temp)@y
    return a

#predict with x_train and a
def predict(x_train,x_test,a,sigma):
    K=kernel(x_train,x_test,sigma)
    a=np.array([a])
    y=a@K
    return y

#calcalute MSE
def cal_mse(y_true,y_train):
     MSE=np.sum(np.power((y_train.reshape(-1,1) - y_true),2))/len(y_train)
     return MSE


#plot a 3D picture for sigma and gamma
def plot_3d(result):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X=result[:,0]
    Y=result[:,1]
    Z=result[:,2]
    ax.plot_trisurf(X,Y,Z)
    ax.set_xlabel('gamma')
    ax.set_ylabel('sigma')
    ax.set_zlabel('cross-validation error')
    plt.title("Question5B")
    plt.show()

#split data for 5-fold cross validation
#k from 0 to 4 to get different data set
def split_data(k):
    number1 = round(data.shape[0] * 1 / 5)
    number2=round(data.shape[0] * 2 / 5)
    number3=round(data.shape[0] * 3 / 5)
    number4=round(data.shape[0] * 4 / 5)
    y_train=0
    x_train=0
    y_test=0
    x_test=0
    if(k==0):
        test, training = data[:number1, :], data[number1:, :]
        y_train = training[:, 13]
        x_train = training[:, :13]
        y_test = test[:, 13]
        x_test = test[:, :13]
    elif(k==1):
        test= data[number1:number2, :]
        training=np.vstack((data[:number1,:],data[number2:,:]))
        y_train = training[:, 13]
        x_train = training[:, :13]
        y_test = test[:, 13]
        x_test = test[:, :13]
    elif(k==2):
        test= data[number2:number3, :]
        training=np.vstack((data[:number2,:],data[number3:,:]))
        y_train = training[:, 13]
        x_train = training[:, :13]
        y_test = test[:, 13]
        x_test = test[:, :13]
    elif(k==3):
        test= data[number3:number4, :]
        training=np.vstack((data[:number3,:],data[number4:,:]))
        y_train = training[:, 13]
        x_train = training[:, :13]
        y_test = test[:, 13]
        x_test = test[:, :13]
    elif(k==4):
        test,training= data[number4:, :],data[:number4,:]
        y_train = training[:, 13]
        x_train = training[:, :13]
        y_test = test[:, 13]
        x_test = test[:, :13]
    return y_train,x_train,y_test,x_test

#QuestionA
def QuestionA():
    #gamma
    gamma = [pow(2, -40), pow(2, -39), pow(2, -38), pow(2, -37),
             pow(2, -36), pow(2, -35), pow(2, -34), pow(2, -33),
             pow(2, -32), pow(2, -31), pow(2, -30), pow(2, -29),
             pow(2, -28), pow(2, -27), pow(2, -26)]

    #sigma
    sigma = [pow(2, 7), pow(2, 7.5), pow(2, 8), pow(2, 8.5), pow(2, 9),
             pow(2, 9.5), pow(2, 10), pow(2, 10.5), pow(2, 11),
             pow(2, 11.5), pow(2, 12), pow(2, 12.5), pow(2, 13)]
    length = len(gamma) * len(sigma)
    res = [[0.0] * 3] * length  #array to store the results
    res = np.array(res)
    for m in range(len(gamma)):
        print("gamma now is " + str(gamma[m]))
        for n in range(len(sigma)):
            print("sigma now is " + str(sigma[n]))
            mse = 0
            for i in range(5):
                # print("loop is "+str(i))
                y_train, x_train, y_test, x_test = split_data(i)
                K = kernel(x_train, x_train, sigma[n])
                alpha = a_star(x_train, y_train, K, gamma[m])
                y = predict(x_train=x_train, x_test=x_test, a=alpha, sigma=sigma[n])
                mse += cal_mse(y, y_test)
                print("mse now is " + str(mse))
            mse_res = mse / 5
            print("avarage mse is " + str(mse_res))
            print("m" + str(m))
            print("n" + str(n))
            number = 13 * m + n
            print("now number is " + str(number))
            res[number][0] = gamma[m]
            res[number][1] = sigma[n]
            res[number][2] = mse_res
            print(res[number])
    np.savetxt("Q5_result.txt", res)
    result = res[res[:, 2].argsort()]
    print(result)
    print("the best gamma is " + str(result[0][0]) +
          "the best sigam is " + str(result[0][1]) +
          "the mse now is " + str(result[0][2]))

#QuestionB
def QuestionB():
    res=np.loadtxt("Q5_result.txt")
    plot_3d(res)

#QuestionC
def QuestionC():
    number=round(data.shape[0] * 2 / 3)
    training, test = data[:number, :], data[number:, :]
    gamma=pow(2,-26)
    sigma=pow(2,13)
    y_train = training[:, 13]
    x_train = training[:, :13]
    y_test = test[:, 13]
    x_test = test[:, :13]
    K = kernel(x_train, x_train, sigma)
    alpha = a_star(x_train, y_train, K, gamma)
    y1 = predict(x_train=x_train, x_test=x_test, a=alpha, sigma=sigma)
    y2 = predict(x_train=x_train,x_test=x_train,a=alpha,sigma=sigma)
    mse_test= cal_mse(y1, y_test)
    mse_train=cal_mse(y2,y_train)
    print("MSE of test set "+str(mse_test))
    print("MSE of training set "+str(mse_train))


#repeat for Question5c over 20 random(2/3,1/3) splits of your data
def QuestionD_4():
    MSE_test=[0.0]*20
    MSE_train=[0.0]*20
    MSE_train=np.array(MSE_train)
    MSE_test=np.array(MSE_test)
    for i in range(20):
        number=round(data.shape[0] * 2 / 3)
        np.random.shuffle(data)
        training, test = data[:number, :], data[number:, :]
        gamma=pow(2,-26)
        sigma=pow(2,13)
        y_train = training[:, 13]
        x_train = training[:, :13]
        y_test = test[:, 13]
        x_test = test[:, :13]
        K = kernel(x_train, x_train, sigma)
        alpha = a_star(x_train, y_train, K, gamma)
        y1 = predict(x_train=x_train, x_test=x_test, a=alpha, sigma=sigma)
        y2 = predict(x_train=x_train,x_test=x_train,a=alpha,sigma=sigma)
        mse_test= cal_mse(y1, y_test)
        mse_train=cal_mse(y2, y_train)
        MSE_test[i]=mse_test
        MSE_train[i]=mse_train
    Mse_train = MSE_train.mean()
    Mse_test = MSE_test.mean()
    STD_train = np.std(MSE_train, ddof=1)
    STD_test = np.std(MSE_test, ddof=1)

    print("The average MSE of training set is " + str(Mse_test))
    print("The average MSE of test set is " + str(Mse_train))
    print("The standard deviarions of train error is " + str(STD_train))
    print("The standard deviarions of test error is " + str(STD_test))
if __name__ == '__main__':
    #QuestionA()
    #QuestionB()
    #QuestionC()
    #QuestionD_4()

