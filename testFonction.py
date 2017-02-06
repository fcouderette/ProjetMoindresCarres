# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:00:22 2017

@author: frederique
"""
import main

import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt

def f(a1,w1,a2,w2,a3,w3,t):
    f1=a1*np.sin(w1*t)+a1*np.cos(w1*t)
    f2=a2*np.sin(w2*t)+a2*np.cos(w2*t)
    f3=a3*np.sin(w3*t)+a3*np.cos(w3*t)
    f=f1+f2+f3
    return f

myTime=np.arange(233)

a10=0.5
a20=0.5
a30=0.5
w10=(2*math.pi)/(6*12)#(2*math.pi)/(6*12)
w20=(2*math.pi)/(12)#(2*math.pi)/(6)
w30=(2*math.pi)/(6)#(2*math.pi)/(12)

obs,res=main.MC("thickness-of-sea.xlsx", a10, a20, a30, w10, w20, w30)
myRes=res #[0]
#print('res = ',res)
#print('obs = ',obs)

a1=myRes[0]
w1=myRes[1]
a2=myRes[2]
w2=myRes[3]
a3=myRes[4]
w3=myRes[5]



y = f(a1,w1,a2,w2,a3,w3,myTime)

plt.figure()
plt.plot(myTime, y, 'r', label='modeled thickness')
plt.plot(myTime, obs, 'b', label='theoric thickness')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('ice thickness (m)')
plt.xlabel("time (per month)")




plt.savefig('modeled_thickness.png')




