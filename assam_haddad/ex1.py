# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import matplotlib.pyplot as plt

def dataGenerator(mu,sigma,nb) :
    
    donnees = np.random.multivariate_normal(mu,sigma,nb)
    return donnees


def printData(data0, data1) :

    X0 = [x for (x,y) in data0]
    Y0 = [y for (x,y) in data0]
    colors = ['r','b']
    plt.plot(X0, Y0, 'x',color=colors[0])
    X1 = [x for (x,y) in data1]
    Y1 = [y for (x,y) in data1]
    plt.plot(X1, Y1, 'x',color=colors[1])
    plt.show()
    

    
#ex1
if __name__ == '__main__' :
    mu0 = np.array([0,0])
    mu1 = np.array([3,2])
    sigma = np.matrix([[1,0.5], [0.5, 1]])
    #Données d'apprentissage
    app_0 = dataGenerator(mu0,sigma,10)
    app_1 = dataGenerator(mu1,sigma,10)
    printData(app_0,app_1)
    #Données tests
    test_0 = dataGenerator(mu0,sigma,1000)
    test_1 = dataGenerator(mu1,sigma,1000)
    printData(test_0,test_1)
    




    
    