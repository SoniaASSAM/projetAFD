# -*- coding: utf-8 -*-

"""
Created on Mon Jan  8 22:21:46 2018
@author: Raouf
"""

# -*- coding: utf-8 -*-

######################## EXERCICE 2 #########################
import ex1
import numpy as np
import sklearn.discriminant_analysis as da
import math as m
from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pds

#Global parameters 
k = 2
mu0 = np.array([0,0])
mu1 = np.array([3,2])
mu = [mu0,mu1]
sigma = np.matrix([[1,0.5], [0.5, 1]])
pi = [1/2,1/2]
n1 = 10
n2 = 10
n = n1 +n2

#DataFrames
classifApp = None
classifTest = None



############ Fonction qui met à jour la moyenne ############
def update_mu() :
    mu[0][0] = classifApp[classifApp.prediction_lda == 0].x1.mean()
    mu[0][1] = classifApp[classifApp.prediction_lda == 0].x2.mean()
    mu[1][0] = classifApp[classifApp.prediction_lda == 1].x1.mean()
    mu[1][1] = classifApp[classifApp.prediction_lda == 1].x2.mean()


############ Fonction qui met à jour pi ############
def update_pi() :
    for i in range(len(pi)) :
        pi[i] = len(classifApp[classifApp['prediction_lda']==i]) / len(classifApp)
        
  ############ Fonction qui met à jour sigma ############
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

      
############ Fonction implémentant Naive Bayes #############
def naive_bayes(x) :
    dk = []
    for i in range(k) :
        dk.append(delta(i,x))
    return np.argmax(dk, axis=0)[0][0]
       
########### Fonction calculant la LDA avec paramètre lambda pour question 5 ############   
def var_LDA(x, lmbda) :
    global sigma
    s = sigma
    sigma = var_sigma(lmbda)
    z = LDA(x)
    sigma = s
    return z


########### Fonction calculant le taux de bonne classificatio ###################
def taux_bonne_classif(df) :
    return {'skn' : len(df[df['class']==df.prediction_skn]) / len(df),
            'lda' : len(df[df['class']==df.prediction_lda]) / len(df)}
    

############ Fonction implémentant l'Analyse Discriminante Linéaire ############
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


######## Sigma pour question 5 ##########
def var_sigma(lmbda) :
    Xapp_0 = classifApp[classifApp['class'] == 0].as_matrix(["x1","x2"])
    Xapp_1 = classifApp[classifApp['class'] == 1].as_matrix(["x1","x2"])
    global n1
    global n2
    n1 = len(Xapp_0)
    n2 = len(Xapp_1)
    n = n1+n2
    sigma_0 = np.cov(Xapp_0.T)
    sigma_1 = np.cov(Xapp_1.T)
    p1 = lmbda * ( n1/n * sigma_0 + n2/n * sigma_1)
    p2 = (1-lmbda) * np.identity(len(sigma))
    return p1+p2



############ Fonction qui print la frontière de décision ############
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

############ Fonction qui calcule le delta ############
def delta(k,x) :
    sigma_inversee = np.linalg.inv(sigma)
    uk = np.matrix(mu[k]).T
    #print(x.T*sigma_inversee*uk -1/2*mu[k]*sigma_inversee*uk)
    p1 = np.transpose(x - 0.5 * uk) * sigma_inversee * uk
    #p1 = x.T*sigma_inversee*uk -1/2*mu[k]*sigma_inversee*uk    
    log = m.log(pi[k])
    
    return p1 + log
  
############ UNITILISE : Fonction qui print la frontière de décision avec une autre méthode############
#Fonctionne pas super bien donc utiisation d'une autre méthode
def decision_boundary(C0, C1, sigma, mu) :
    w = np.dot(np.linalg.inv(sigma), np.transpose(mu[0]-mu[1]))
    b = -0.5 * (np.transpose(mu[0]-mu[1]).dot(np.linalg.inv(sigma).dot(mu[0]+mu[1]))) + m.log(pi[0]/pi[1])

    Y0 = [y for (x, y) in C0]
    Y1 = [y for (x, y) in C1]

    y0 = min(min(Y0), min(Y1))
    y1 = max(max(Y0), max(Y1))
    
    x0 = (-b-w[1]*y0) / w[0]
    x1 = (-b-w[1]*y1) / w[0]

    P1 = [x0, y0]
    P2 = [x1, y1]
    
    ##Print
    
    X0 = [x for (x, y) in C0] # On récupère les abcisses
    YY0 = [y for (x, y) in C0] # On récupère les ordonnées
    plt.plot(X0, YY0, '+',  color="red")
    
    X1 = [x for (x, y) in C1]
    YY1 = [y for (x, y) in C1]
    plt.plot(X1, YY1, '+', color="blue")
    
    plt.plot((P1[0], P2[0]), (P1[1], P2[1]), color="black")
    plt.show()

#DataFrames
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

#DataFrames
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


#Fonction applicant LDA de la librairie Sklearn
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


############ MAIN ############
if __name__ == '__main__' :
    
    
    ############## Question 1 ####################
    print("##Question 1 : \n")
    init_classifApp()
    init_classifTest()
    res_impl = LDA(Xtest)
    res_skn =skn(Xtest)
    classifApp['prediction_skn'] = skn(classifApp.as_matrix(['x1','x2']))
    
    #Print results
    print(classifApp)
    print("\n - Taux de bonne clasification : "+
          str(taux_bonne_classif(classifApp)))
    
    print("\n - Résultats identitques ? "
          +str(np.array_equal(res_skn,res_impl))) ##Retourne booléen
    
    print("\n - Frontiere décision Q1")
    decision_boundary(Xapp_0,Xapp_1, sigma, mu)
    decision_boundary(Xtest_0,Xtest_1, sigma,mu)
    
    ############## Question 2 & 3 ####################
    print("\n\n\n ##Question 2 : ")
   
    Xapp_0 = ex1.dataGenerator(mu[0],sigma,10)
    Xapp_1 = ex1.dataGenerator(mu[1],sigma,10)
    Xapp = np.concatenate((Xapp_0,Xapp_1))
    Xapp_0[0] = [-10,-10]
    Xapp = np.concatenate((Xapp_0,Xapp_1))
    init_classifApp()
    
    #MAJ données de test après changement de la donnée d'app
    update_mu()
    update_sigma()
    pi = [1/2,1/2]
    Xtest_0 = ex1.dataGenerator(mu[0],sigma,1000)
    Xtest_1 = ex1.dataGenerator(mu[1],sigma,1000)
    Xtest = np.concatenate((Xtest_0,Xtest_1))

    init_classifTest()
    res_impl2 = LDA(Xtest)
    res_skn2 = skn(Xtest)
    classifApp.loc[20:,['prediction_skn']] = res_skn2
    
    #Print results
    print("\n - Résultats après modification de la première observation de la classe C0 : ")
    print("\n - Taux de bonne clasification : "+
          str(taux_bonne_classif(classifApp)))
    
    print("\n - Résultats identitques ? "
          +str(np.array_equal(res_skn2,res_impl2)))
     
    print("\n - Frontière de décision Q2")
    print_decision_boundary(Xapp_0,Xapp_1)
    print_decision_boundary(Xtest_0,Xtest_1)

    
    #Question4
    #Faire éloigner la cov mettre exemple 10 5 5 10 => On aura des taux de bonnes classif à 60%
    
    #Question 5
    #Pour lambda = 1 On se retrouve avec les parametres de LDA
    
    print("\n\n\n##Question 5 : comparaison LDA & Nouvelle formule avec lambda = 1")
    mu0 = np.array([0,0])
    mu1 = np.array([3,2])
    mu = [mu0,mu1]
    sigma = np.matrix([[1,0.5], [0.5, 1]])
    pi = [1/2,1/2]
    Xapp_0 = ex1.dataGenerator(mu[0],sigma,10)
    Xapp_1 = ex1.dataGenerator(mu[1],sigma,10)
    Xapp = np.concatenate((Xapp_0,Xapp_1))
    Xtest_0 = ex1.dataGenerator(mu[0],sigma,1000)
    Xtest_1 = ex1.dataGenerator(mu[1],sigma,1000)
    Xtest = np.concatenate((Xtest_0,Xtest_1))
    init_classifApp()
    init_classifTest()
    
    print("\nTracer la courbe des taux en fonction des lambdas")
    print("\n\t## ATTENTION ## Traitement assez lent (1 à 2m), pensez à prendre un café avec vous")
    
    taux = []
    pas = 0.1
    lmbdas = []
    #Données d'apprentissage
    app_0 = ex1.dataGenerator(mu[0],sigma,10)
    app_1 = ex1.dataGenerator(mu[1],sigma,10)
    app = np.concatenate((app_0,app_1))
    tst_0 = ex1.dataGenerator(mu[0],sigma,1000)
    tst_1 = ex1.dataGenerator(mu[1],sigma,1000)
    tst = np.concatenate((tst_0,tst_1))
    for i in range(11) :
        print(i)
        Xapp = app
        Xtest = tst
        mu0 = np.array([0,0])
        mu1 = np.array([3,2])
        mu = [mu0,mu1]
        sigma = np.matrix([[1,0.5], [0.5, 1]])
        pi = [1/2,1/2]
        init_classifApp()
        init_classifTest()
        lmbdas.append(i*pas)
        var_LDA(Xtest,i*pas)
        taux.append(taux_bonne_classif(classifApp)['lda'])
    plt.plot(lmbdas,taux)
    plt.title("Taux de bonne classification en fonction de lambdas")
    plt.show()