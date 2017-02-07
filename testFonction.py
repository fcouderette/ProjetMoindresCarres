# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:00:22 2017

@author: frederique
"""
import main2

import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt

def f(a1,w1,a2,t):
    f1=a1*np.sin(w1*t)
    f2=0 #a2*t
    f=f1+f2
    return f



a10=500
w10=(2*math.pi)/(4)
a20=0 #0.00000000001

obs,res=main2.MC("ground-temperature-in-permafrost.xlsx", a10, a20, w10)
myRes=res #[0]

myTime=np.arange(obs.shape[0])
#print('res = ',res)
#print('obs = ',obs)

a1=myRes[0]
print('a1 = ',a1)
w1=myRes[1]
print('w1 = ',w1)
a2=myRes[2]
print('a2 = ',a2)
#w2=myRes[3]
#a3=myRes[4]
#w3=myRes[5]



y = f(a1,w1,a2,myTime)

plt.figure()
plt.plot(myTime, y, 'r', label='modeled temperature')
plt.plot(myTime, obs, 'b', label='theoric temperature')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('permafrost temperature (degrees)')
plt.xlabel("time (per day)")




plt.savefig('permafrost_temperature.png')




