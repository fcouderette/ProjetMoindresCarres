# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:18:08 2016

@author: frederique
"""


import numpy as np
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


if __name__=='__main__':
    
    print("\n*** STEP 1 : IMPORTATION OF DATA ***\n")
    mylist=importData("thickness-of-sea.xlsx")
    
    print("\n*** STEP 2 : OBSERVATIONS ***\n")
    # Transforms list into vector
    myVector=np.asarray(mylist)
    nb=len(myVector)
    obs=myVector.reshape(nb,1)
    print("nb of observations : \n",nb, "\n")
    print("vecteur myVector : \n",myVector, "\n")
    print("vecteur obs : \n",obs, "\n")
    
    a10=2
    a20=2
    a30=2
    w10=(2*math.pi)/(6*12)
    w20=(2*math.pi)/(6)
    w30=(2*math.pi)/(12)
    
    myTime=np.arange(obs.shape[0])
    #print("vecteur temps : \n",myTime, "\n")
    
    print("\n*** STEP 3 : REDUCED OBSERVATIONS ***\n")
    redObs=np.zeros(obs.shape[0]).reshape(obs.shape[0],1)
    for i in range(obs.shape[0]):
        redObs[i]=obs[i]-function(a10,w10,a20,w20,a30,w30,myTime[i])
        
    print("vecteur obs réduites : \n",redObs, "\n")
        










