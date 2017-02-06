# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:18:08 2016

@author: frederique
"""


import numpy as np
from numpy.linalg import inv
import math
from xlrd import open_workbook

def importData(filepath):
    # Opens file
    wb = open_workbook(filepath)
    
    # Creates list that will contain values
    mylist=[]
    
    # For each sheet of xslx file
    for thesheet in wb.sheets():
        values = []
        # For each line
        for row in range(thesheet.nrows):
            col_value = []
            # If wanted line
            if (row==3):
                # Opens/Creates text file
                fichier = open("monbeaupetitfichier.txt", "w")
                # For eah column
                for col in range(thesheet.ncols): 
                    # Gets value
                    value  = (thesheet.cell(row,col).value) 
                    # Sets float value to integer
                    try : value = str(float(value))
                    except : pass
                    # Writes value in text file
                    col_value.append(value)
                    fichier.write(value)  
                    fichier.write("\n") 
                    # If column only about float values
                    if col>2:
                        # Adds value to list
                        mylist.append(float(value))
                    values.append(col_value)
                # Closes text file
                fichier.close()
    return mylist

def function(a1,w1,a2,w2,a3,w3,t):
    member1=a1*math.sin(w1*t) + a1*math.cos(w1*t)
    member2=a1*math.sin(w1*t) + a2*math.cos(w2*t)
    member3=a1*math.sin(w1*t) + a3*math.cos(w3*t)
    f=member1+member2+member3
    return f
    
    
def ComputeReducedObs(vectorObs, vectorTime, a10,a20,a30,w10,w20,w30):
    redObs=np.zeros(vectorObs.shape[0]).reshape(vectorObs.shape[0],1)
    for i in range(vectorObs.shape[0]):
        redObs[i]=vectorObs[i]-function(a10,w10,a20,w20,a30,w30,vectorTime[i])
    return redObs
    

def computeVarCovarMatrix(prec, nl):
    VarCovarMatrix=(1/(prec**2)) * np.eye(nl)
    return VarCovarMatrix
    
def computeAmatrix(vectorObs, vectorTime, nb_param,  a10,a20,a30,w10,w20,w30):
    A = np.ones((vectorTime.shape[0],nb_param))
    for i in range (vectorTime.shape[0]):
        A[i][0] = np.sin(w10*vectorTime[i])+np.cos(w10*vectorTime[i])
        A[i][1] = a10*w10*np.cos(w10*vectorTime[i])-a10*w10*np.sin(w10*vectorTime[i])
        A[i][2] = np.sin(w20*vectorTime[i])+np.cos(w20*vectorTime[i])
        A[i][3] = a20*w20*np.cos(w20*vectorTime[i])-a20*w20*np.sin(w20*vectorTime[i])
        A[i][4] = np.sin(w30*vectorTime[i])+np.cos(w30*vectorTime[i])
        A[i][5] = a30*w30*np.cos(w30*vectorTime[i])-a30*w30*np.sin(w30*vectorTime[i])
    print("\nA =\n",A)
    return A
    
    


def MC(datapath, a10, a20, a30, w10, w20, w30):
    print("\n*** STEP 1 : IMPORTATION OF DATA ***\n")
    mylist=importData(datapath)
    nbUnknown=6
    
    print("\n*** STEP 2 : OBSERVATIONS ***\n")
    # Transforms list into vector
    myVector=np.asarray(mylist)
    nb=len(myVector)
    obs=myVector.reshape(nb,1)
    print("nb of observations : \n",nb, "\n")
    #print("vecteur myVector : \n",myVector, "\n")
    #print("vecteur obs : \n",obs, "\n")
    
    # Vector Time (per month)
    myTime=np.arange(obs.shape[0])
    #print("vecteur temps : \n",myTime, "\n")
    
    print("\n*** STEP 3 : REDUCED OBSERVATIONS ***\n")
    redObs=ComputeReducedObs(obs, myTime, a10, a20, a30, w10, w20, w30)
    print("vecteur obs réduites : \n",redObs, "\n")
    
    print("\n*** STEP 4 : A ***\n")
    A=computeAmatrix(obs, myTime, nbUnknown, a10,a20,a30,w10,w20,w30)
    
    
    
    # Variance-covariance matrix and weight matrix
    print("\n*** STEP 5 : WEIGHT MATRIX ***\n")
    VarCovarMatrix=computeVarCovarMatrix(0.05, obs.shape[0])
    WeightMatrix=inv(VarCovarMatrix)
    print("\nWeight Matrix =\n",WeightMatrix)

    
    # Normal Matrix
    print("\n*** STEP 6 : NORMAL MATRIX ***\n")
    NormalMatrix=np.dot(np.dot(A.T,WeightMatrix),A)
    print("\nNormal Matrix =\n",NormalMatrix)
    
    # Matricial Computing
    print("\n*** STEP 7 : MATRICIAL COMPUTING ***\n")
    deltaParameters=inv(NormalMatrix).dot(A.T.dot(WeightMatrix).dot(redObs))
    print("\nUnknown delta parameters =\n",deltaParameters)
    
    # Determination des parametres inconnus
    print("\n*** STEP 8 : UNKNOWN PARAMETERS ***\n")
    vectorParameters=deltaParameters+np.array([a10,a20,a30,w10,w20,w30]).reshape(nbUnknown,1)
    print("\nD'où les paramètres :\n",vectorParameters)
    
    # Residuals determination 
    print("\n*** STEP 9 : RESIDUAL DETERMINATION ***\n")
    vectorResiduals = redObs-np.dot(A,deltaParameters)
    print("\nD'où les résidus :\n",vectorResiduals)
    
    print("\n*** STEP 10 : PRECISION OF PARAMETERS ***\n")
    Qx = inv(np.dot(np.dot(A.T,WeightMatrix),A))
    print("\nD'où Qx =\n",Qx)
    
    print("\n*** STEP 11 : PRECISION OF RESIDUALS ***\n")
    Qv = VarCovarMatrix-np.dot(np.dot(A,Qx),A.T)
    print("\nD'où Qv =\n",Qv)    
    
    
    
    
    return obs,vectorParameters #vectorResiduals, Qx, Qv
    




if __name__=='__main__':
    
    a10=2
    a20=2
    a30=2
    w10=(2*math.pi)/(6*12)
    w20=(2*math.pi)/(6)
    w30=(2*math.pi)/(12)
    
    res=MC("thickness-of-sea.xlsx", a10, a20, a30, w10, w20, w30)
    
    
        










