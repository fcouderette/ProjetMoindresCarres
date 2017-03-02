# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:14:28 2017

@author: charlotte
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint


def importData(file_path):
    """Import the two needed vectors"""   
    with open(file_path,'r') as mon_fichier:
        vectorTime = []
        vectorObs = []
        for line in mon_fichier:
           split = line.split()
           vectorTime.append(int(split[0]))
           h=split[1]
           transfo = float(h[0]) + float(h[1:])/60
           vectorObs.append(transfo)
        vectorObs = np.array(vectorObs)
        vectorTime = np.array(vectorTime)   
           
    mon_fichier.close()
    return vectorTime,vectorObs

    
def f(t,a,w,b,c,d):
    y = a*np.sin(w*t + d)+b*t + c
    return y
 
def ComputeReducedObs(vectorObs, vectorTime,a,w,b,c,d):   
    redObs=np.zeros(vectorTime.shape[0])
    for i in range(vectorTime.shape[0]):
        redObs[i]=vectorObs[i] - f(vectorTime[i],a,w,b,c,d)
    return redObs
    
def computeAmatrix(vectorObs, vectorTime,a,w,b,c,d):
    A = np.zeros((vectorTime.shape[0],5))
    for i in range (vectorTime.shape[0]):
        A[i][0] = np.sin(w * vectorTime[i]+d)  #--- a
        A[i][1] = a * vectorTime[i] * np.cos(w * vectorTime[i]+d)  #--- w
        A[i][2] = vectorTime[i]  #--- b
        A[i][3] = 1  #--- c
        A[i][4] = a * np.cos(w * vectorTime[i]+d)
    #print("\nA =\n",A)
    return A

def computeVarCovarMatrix(prec, nl):
    """Construction of the variance/covariance matrix"""
    VarCovarMatrix=(1/(prec**2)) * np.eye(nl)
    return VarCovarMatrix


def MC(vectorObs,vectorTime,a0,w0,b0,c0,d0):
    
    a=a0
    w=w0
    b=b0
    c=c0
    d=d0
    
#    print("\n*** STEP 1 : IMPORTATION OF DATA ***\n")
#    vectorObs,vectorTime = importData(datapath)   
    
    print("\n*** STEP 2 : OBSERVATIONS ***\n")
    nb = vectorObs.shape[0]
    print("nb of observations : \n",nb, "\n")
    
    sigma0=1
    ds=0.00000001
    n = 0
    while (np.abs(ds)<=1):
        print("\n =====> ITERATION ",n)
        print("\nsigma0 : ", sigma0)
        print("\n*** STEP 3 : REDUCED OBSERVATIONS ***\n")
        redObs=ComputeReducedObs(vectorObs, vectorTime, a,w,b,c,d)
        #print("vecteur obs réduites : \n",redObs, "\n")
        
        print("\n*** STEP 4 : A ***\n")
        A=computeAmatrix(vectorObs, vectorTime, a,w,b,c,d)
        
        # Variance-covariance matrix and weight matrix
        print("\n*** STEP 5 : WEIGHT MATRIX ***\n")
        VarCovarMatrix=computeVarCovarMatrix(sigma0, vectorObs.shape[0])
        WeightMatrix=np.linalg.inv(VarCovarMatrix)
        #print("\nWeight Matrix =\n",WeightMatrix)
        
        # Normal Matrix
        print("\n*** STEP 6 : NORMAL MATRIX ***\n")
        NormalMatrix=np.dot(np.dot(A.T,WeightMatrix),A)
        #print("\nNormal Matrix =\n",NormalMatrix)
        
        # Matricial Computing
        print("\n*** STEP 7 : MATRICIAL COMPUTING ***\n")
        deltaParameters=np.linalg.inv(NormalMatrix).dot(A.T.dot(WeightMatrix).dot(redObs))
        #print("\nUnknown delta parameters =\n",deltaParameters)
        
        # Determination des parametres inconnus
        print("\n*** STEP 8 : UNKNOWN PARAMETERS ***\n")
        tab=np.array([a,w,b,c,d])
        vectorParameters = np.zeros(tab.shape[0])    
        for i in range(tab.shape[0]):
            vectorParameters[i] = tab[i]+deltaParameters[i]
        #vectorParameters=tab+deltaParameters
        print("\nD'où les paramètres :\n",vectorParameters)
        
        a = vectorParameters[0]
        w = vectorParameters[1]
        b = vectorParameters[2]
        c = vectorParameters[3] 
        d = vectorParameters[4]
        n=n+1
        
        # Residuals determination 
        print("\n*** STEP 9 : RESIDUAL DETERMINATION ***\n")
        vectorResiduals = redObs-np.dot(A,deltaParameters)
        #print("\nD'où les résidus :\n",vectorResiduals)
        
        # sigma02
        sigma0_2=(vectorResiduals.T.dot(WeightMatrix.dot(vectorResiduals)))/(nb-5)
        #print("\n sigma0_2/(sigma0*sigma0) = ", sigma0_2/(sigma0*sigma0))
        
        ds=sigma0_2-(sigma0*sigma0)
        print("\n ds = ", ds)
        a0=a
        w0=w
        b0=b
        c0=c
        d0=d
        
        
    
    print("\n*** STEP 10 : PRECISION OF PARAMETERS ***\n")
    Qx = np.linalg.inv(np.dot(np.dot(A.T,WeightMatrix),A))
    #print("\nD'où Qx =\n",Qx)
 
    print("\n*** STEP 11 : PRECISION OF RESIDUALS ***\n")
    Qv = VarCovarMatrix-np.dot(np.dot(A,Qx),A.T)
    #print("\nD'où Qv =\n",Qv)
    
     
    print("\n*** STEP 12 : FIABILITY ESTIMATORS ***\n")
    # Fiability Estimator
    prod=np.dot(Qv,WeightMatrix)
    zi=np.zeros((nb,nb))
    myList=np.zeros((nb,1))

    for i in range(nb):
        zi[i,i]=prod[i,i]
        myList[i]=prod[i,i]
    print("zi : \n",zi)
    #print("\nlist of zi : \n",myList)
    print("Sum of zi =",np.sum(myList))
    

    return vectorObs,vectorParameters #vectorResiduals, Qx, Qv   
    



def RANSAC(vectorObs,vectorTime,a,w,b,c,d,nb):
    
    # Observation vector
    vectorObsRed = np.zeros(nb)
    # Time vector
    vectorTimeRed = np.zeros(nb)
    
    for i in range (nb):
        indice = randint(0,vectorTime.shape[0]-1)  
        vectorObsRed[i] = vectorObs[indice]
        vectorTimeRed[i] = vectorTime[indice]
        
    vectorObs,vectorParameters = MC(vectorObs,vectorTime,a,w,b,c,d)
    
    return vectorObs,vectorParameters
 

if __name__=='__main__':

    vectorTime,vectorObs = importData("heure.res")
    
    a = 0.2
    w = 2*math.pi*(1/160)
    b = -0.0005
    c = 6  
    d = 0
    
    vectorObs,vectorParameters = MC(vectorObs,vectorTime,a,w,b,c,d)
    y=np.zeros(vectorTime.shape[0])
    for i in range(vectorTime.shape[0]):
        nb = f(vectorTime[i],vectorParameters[0],vectorParameters[1],vectorParameters[2],vectorParameters[3],vectorParameters[4])  
        y[i] = nb


    plt.figure()
    plt.plot(vectorTime,vectorObs,'b', label='theoric time')
    
    plt.plot(vectorTime,y,'r', label='modeled time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('sunset time')
    plt.xlabel("day in the year")

    plt.show()

    vectorObs,vectorParameters = RANSAC(vectorObs,vectorTime,a,w,b,c,d,90)









#a*cos(w*t) + b*cos(w2*t)
#a*cos(w*t) -bt +C
#a=0.1


