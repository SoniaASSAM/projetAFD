# -*- coding: utf-8 -*-

"""
Created on Mon Jan  8 22:21:46 2018
@author: Raouf
"""


# -*- coding: utf-8 -*-

import ex1
import numpy as np
import sklearn.discriminant_analysis as da
import math
from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pds

#Les paramètres
k = 2
mu0 = np.array([0,0])
mu1 = np.array([3,2])
mu = [mu0,mu1]
sigma = np.matrix([[1,0.5], [0.5, 1]])
pi = [1/2,1/2]
n1 = 10
n2 = 10
n = n1 +n2

classifApp = None
classifTest = None



    
def update_mu() :
    mu[0][0] = classifApp[classifApp.prediction_lda == 0].x1.mean()
    mu[0][1] = classifApp[classifApp.prediction_lda == 0].x2.mean()
    mu[1][0] = classifApp[classifApp.prediction_lda == 1].x1.mean()
    mu[1][1] = classifApp[classifApp.prediction_lda == 1].x2.mean()
    
 
def naive_bayes(x) :
    dk = []
    for i in range(k) :
        dk.append(delta(i,x))
    return np.argmax(dk, axis=0)[0][0]


def update_pi() :
    for i in range(len(pi)) :
        pi[i] = len(classifApp[classifApp['prediction_lda']==i]) / len(classifApp)
       
        
def var_LDA(x, lmbda) :
    global sigma
    s = sigma
    sigma = var_sigma(lmbda)
    z = LDA(x)
    sigma = s
    return z

 
def taux_bonne_classif(df) :
    return {'skn' : len(df[df['class']==df.prediction_skn]) / len(df),
            'lda' : len(df[df['class']==df.prediction_lda]) / len(df)}



def update_sigma() :
    
    global sigma
    sigma_0 = classifApp[classifApp.prediction_lda == 0].as_matrix(['x1','x2'])
    sigma_1 = classifApp[classifApp.prediction_lda == 1].as_matrix(['x1','x2'])
    sigma_0 = np.cov(sigma_0.T)
    sigma_1 = np.cov(sigma_1.T)
    n0 = len(classifApp[classifApp.prediction_lda == 0])
    n1 = len(classifApp[classifApp.prediction_lda == 1])
    n = n0 + n1
    sigma = np.add(n0/n * sigma_0, n1/n * sigma_1)
    


def LDA(X) :

    global classifApp
    res = []
    global Xapp
    update_mu()
    update_sigma()
    for i in range(len(X)) : #Données de tests à classer par rapport aux données d'apprentissage
        Xapp = classifApp.as_matrix(['x1','x2'])
        update_pi()
        c = naive_bayes(X[i])
        res.append(c)
        classifTest.loc[i,['x1','x2']] = X[i][0], X[i][1]
        classifTest.loc[i, 'prediction_lda'] = c
        classifApp = classifApp.append(classifTest.iloc[i], ignore_index=True)
    return res


def var_sigma(lmbda) :
    p1 = lmbda * ( n1/n * sigma + n2/n * sigma)
    p2 = (1-lmbda) * np.identity(len(sigma))
    return p1+p2


def print_decision_boundary(X0,X1) :
    plt.scatter(X0[:,0],X0[:,1], c='red')
    plt.scatter(X1[:,0],X1[:,1], c= 'blue')
    X = np.concatenate((X0,X1))
    clf = svm.SVC(kernel = 'linear')
    y = np.hstack((np.zeros(len(X0)), np.ones(len(X1))))
    clf.fit(X, y)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, 'k-')
    plt.show()

 
def delta(k,x) :
    sigma_inversee = np.linalg.inv(sigma)
    uk = np.matrix(mu[k]).T
    #print(x.T*sigma_inversee*uk -1/2*mu[k]*sigma_inversee*uk)
    p1 = np.transpose(x - 0.5 * uk) * sigma_inversee * uk
    #p1 = x.T*sigma_inversee*uk -1/2*mu[k]*sigma_inversee*uk    
    log = math.log(pi[k])
    
    return p1 + log
  

def decision_boundary(sigma,mu) :
    sigma_inversee = np.linalg.inv(sigma)
    u = [i.reshape(1,2).T for i in mu]
    w = sigma_inversee * (u[0] - u[1])
    b = (u[0] - u[1]).T * sigma_inversee * (u[0] + u[1]) / 2 - math.log(pi[0]/pi[1])
    xx = [0,-b/w[0]]
    yy = [-b/w[1],0]
    plt.plot(xx, yy, 'r-')
    plt.show()


def init_classifApp() :
    global classifApp
    classifApp0 = pds.DataFrame()
    classifApp1 = pds.DataFrame()
    classifApp0['class'] = np.zeros(10)
    classifApp1['class'] = np.ones(10)
    
    classifApp0['x1'] = classifApp0.apply(lambda x : Xapp_0[x.name][0], axis=1)
    classifApp0['x2'] = classifApp0.apply(lambda x : Xapp_0[x.name][1], axis=1)
    classifApp1['x1'] = classifApp1.apply(lambda x : Xapp_1[x.name][0], axis=1)
    classifApp1['x2'] = classifApp1.apply(lambda x : Xapp_1[x.name][1], axis=1)
    classifApp = classifApp0.append(classifApp1, ignore_index=True)
    classifApp['prediction_lda'] = [naive_bayes(i) for i in Xapp]
    classifApp['prediction_skn'] = None
    
def init_classifTest() :
    global classifTest
    classifTest0 = pds.DataFrame()
    classifTest1 = pds.DataFrame()
    classifTest0['class'] = np.zeros(1000)
    classifTest1['class'] = np.ones(1000)
    classifTest0['x1'] = classifTest0.apply(lambda x : Xtest_0[x.name][0], axis=1)
    classifTest0['x2'] = classifTest0.apply(lambda x : Xtest_0[x.name][1], axis=1)
    classifTest1['x1'] = classifTest0.apply(lambda x : Xtest_1[x.name][0], axis=1)
    classifTest1['x2'] = classifTest0.apply(lambda x : Xtest_1[x.name][1], axis=1)
    classifTest = classifTest0.append(classifTest1, ignore_index=True)

    
#Données d'apprentissage
Xapp_0 = ex1.dataGenerator(mu[0],sigma,10)
Xapp_1 = ex1.dataGenerator(mu[1],sigma,10)
Xapp = np.concatenate((Xapp_0,Xapp_1))



def skn(X) :
    y = np.hstack((np.zeros(10), np.ones(10)))
    clf = da.LinearDiscriminantAnalysis()
    clf.fit(Xapp[:20],y)
    res = clf.predict(X)
    return res
    


#Données de test
Xtest_0 = ex1.dataGenerator(mu[0],sigma,1000)
Xtest_1 = ex1.dataGenerator(mu[1],sigma,1000)
Xtest = np.concatenate((Xtest_0,Xtest_1))


if __name__ == '__main__' :
    #Question 1
    
    print("Question 1 : ")
    init_classifApp()
    init_classifTest()
    print("Variable à prédire : Données tests")
    res_impl = LDA(Xtest)
    res_skn =skn(Xtest)
    classifApp['prediction_skn'] = skn(classifApp.as_matrix(['x1','x2']))
    print("Résultats identitques ? "
          +str(np.array_equal(res_skn,res_impl)))
    #Résultats différents
    print("Taux de bonne clasification : "+
          str(taux_bonne_classif(classifApp)))
    #print("Frontiere décision avant")
    #print_decision_boundary(Xapp_0,Xapp_1)
    #Question 2
    print("Question 2 : ")
    Xapp_0[0] = [-10,-10]
    Xapp = np.concatenate((Xapp_0,Xapp_1))
    init_classifApp()
    init_classifTest()
    print("Résultats après modif des données app")
    res_impl2 = LDA(Xtest)
    res_skn2 = skn(Xtest)
    classifApp.loc[20:,['prediction_skn']] = res_skn2
    print("Résultats identitques ? "
          +str(np.array_equal(res_skn2,res_impl2)))
    print("Taux de bonne clasification : "+
          str(taux_bonne_classif(classifApp)))
    #print(classifApp)
    #print("frontiere apres")
    #print_decision_boundary(Xapp_0,Xapp_1)
    #Même résultats que précedents ?
    #TODO Pourquoi ?
    #Question3
    print("Question 3 : ")
    print("Frontière de décision: ")
    print_decision_boundary(Xapp_0,Xapp_1)
    #TODO Question4
    #Faire éloigner la cov mettre exemple 10 5 5 10 => On aura des taux de bonnes classif à 60%
    #Question 5
    #Pour lambda = 1 On se retrouve avec les parametres de LDA
    print("Question 5 :")
    print(np.array_equal(var_LDA(Xtest,0),res_impl2))