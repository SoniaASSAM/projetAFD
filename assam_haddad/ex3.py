# -*- coding: utf-8 -*-

############## EXERCICE 3 #########################
import ex1, ex2
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pds
import sklearn.discriminant_analysis as da


########## Question 1 : Choix de classifieurs ############################
#1 - SVM : Support Machine Vector 
#2 - Neuronal Network


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


#Données d'apprentissage
Xapp_0 = ex1.dataGenerator(mu[0],sigma,10)
Xapp_1 = ex1.dataGenerator(mu[1],sigma,10)
Xapp = np.concatenate((Xapp_0,Xapp_1))


#Données de test
Xtest_0 = ex1.dataGenerator(mu[0],sigma,1000)
Xtest_1 = ex1.dataGenerator(mu[1],sigma,1000)
Xtest = np.concatenate((Xtest_0,Xtest_1))

classif = None

#DataFrames 
def init_classif(X0,X1) :

    global classif
    classif0 = pds.DataFrame()
    classif1 = pds.DataFrame()
    classif0['class'] = np.zeros(len(X0))
    classif1['class'] = np.ones(len(X1))
    classif0['x1'] = classif0.apply(lambda x : X0[x.name][0], axis=1)
    classif0['x2'] = classif0.apply(lambda x : X0[x.name][1], axis=1)
    classif1['x1'] = classif1.apply(lambda x : X1[x.name][0], axis=1)
    classif1['x2'] = classif1.apply(lambda x : X1[x.name][1], axis=1)
    classif = classif0.append(classif1)
    classif = classif.sample(frac=1).reset_index(drop=True)

#Fonction classification avec méthode SVM
def testSVM(X) :
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(Xapp,y)
#    w = clf.coef_[0]
#    a = -w[0] / w[1]
#    xx = np.linspace(-5, 5)
#    yy = a * xx - (clf.intercept_[0]) / w[1]
#    plt.plot(xx, yy, 'k-')
#    plt.show()
    return clf.predict(X)
    
#Fonction classification avec méthode de Réseau de Neurones  
def test_neuronal_network(X) :
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(Xapp, y) 
    return clf.predict(X)

#Fonction classification avec méthode d'Analyse Linéaire Discriminante 
def test_lda(X) :
    clf = da.LinearDiscriminantAnalysis()
    clf.fit(Xapp,y)
    return clf.predict(X)

#Fonction calculant le taux de bonne classificatio 
def taux_bonne_classif(classif) :
    n = len(classif)
    return {'lda' : len(classif[classif['class'] == classif.prediction_lda])/n,
            'svm' : len(classif[classif['class'] == classif.prediction_svm])/n,
            'neuronal_net' : len(classif[classif['class'] == classif.prediction_neuronal_net])/n}

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
    
###################### QUESTION 2 : USING RANDOM DATA #####################
    
init_classif(Xtest_0, Xtest_1)
y = np.hstack((np.zeros(len(Xapp_0)), np.ones(len(Xapp_1))))
Xtest = classif.as_matrix(['x1','x2'])
classif['prediction_svm'] = testSVM(Xtest)

classif['prediction_neuronal_net'] = test_neuronal_network(Xtest)

classif['prediction_lda'] = test_lda(Xtest)

#Print results
print("## Résultats de classification avec données synthétiques : \n")
print(taux_bonne_classif(classif))

###################### QUESTION 3 : USING EXTERNAL DATA #####################

#Load data
jdd = pds.DataFrame.from_csv("haberman.csv", header=None).reset_index()
n = len(jdd)
napp = int(0.8 * n)
jdd = jdd.rename(columns={0:'x1', 1:"x2",2:'x3',3:'class'})

#Données d'apprentissage
Xapp = jdd.loc[:napp]#.as_matrix(['x1','x2','x3'])
Xapp_0 = Xapp[Xapp['class'] == 2].as_matrix(['x1','x2','x3'])
Xapp_1 = Xapp[Xapp['class'] == 1].as_matrix(['x1','x2','x3'])

#Données de test
Xtest = jdd.loc[napp:]
Xtest_0 = Xtest[Xtest['class'] == 2].as_matrix(['x1','x2','x3'])
Xtest_1 = Xtest[Xtest['class'] == 1].as_matrix(['x1','x2','x3'])

#Application apprentissage
y = np.hstack((np.zeros(len(Xapp_0)), np.ones(len(Xapp_1))))
Xapp = np.concatenate((Xapp_0,Xapp_1))
init_classif(Xtest_0,Xtest_1)
Xtest = np.concatenate((Xtest_0,Xtest_1))
classif['prediction_svm'] = testSVM(Xtest)
classif['prediction_neuronal_net'] = test_neuronal_network(Xtest)
classif['prediction_lda'] = test_lda(Xtest)

#Print results
print("\n\n## Résultats de classification avec données externes : \n")
print(taux_bonne_classif(classif))
