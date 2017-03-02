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

def f(a1,a2,w,a3,t):
#    f1=a1*np.sin(w1*t)
#    f2=a2*np.log(w2*(t+1))+a3
#    f=(f1+f2)
    #f=a1*np.sin(w1*t)+a2*np.log(t+1)-7
    f=a1*t+a2*(np.sin((t/(1.2*w)))/(t+1))+a3
    return f

# (cos(x)+x)*(1/2)-6

#a10=-12
#a20=281
#a30=235 #6.77e-1
##a*x+b*sin(c/x)
#w10=-4
#w20=13575

#
# 0.005*x+100*sin(x/50)/(x)-6.5
# a1*x+a2*(sin((x/w))/x)+a3

a10=0.00015
a20=-71.87 #3000
w=187.17 #200
a30=-6 #-6.5

obs,res=main2.MC("ground-temperature-in-permafrost.xlsx", a10, a20,a30, w)
myRes=res #[0]

myTime=np.arange(obs.shape[0])
#print('res = ',res)
#print('obs = ',obs)

a1=myRes[0]
print('a1 = ',a1)
a2=myRes[1]
print('a2 = ',a2)
w=myRes[2]
print('w = ',w)
a3=myRes[3]
print('a3 = ',a3)

#w2=myRes[3]
#a3=myRes[4]
#w3=myRes[5]




#a1=1/10
#a2=1/5
#a3=-7
#
#w1=(2*math.pi)/3000
#w2=(2*math.pi)/3000

a=3 #useless
y = f(a1,a2,w,a3,myTime)
#print('\nmyTime :', myTime)

plt.figure()
plt.plot(myTime, y, 'r', label='modeled temperature')
plt.plot(myTime, obs, 'b', label='theoric temperature')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('permafrost temperature (degrees)')
plt.xlabel("time (per day)")




plt.savefig('permafrost_temperature.png')




