import numpy as np
import math
from matplotlib import pyplot as plt
import timeit
x1=([1,7,8,9,4,8,4,2,3,2,7,1,5,2])
x2=([6,2,9,9,8,5,4,1,3,4,1,3,2,7])
yd=([1,1,1,1,1,1,1,0,0,0,0,0,0,0])
for i in range(0,7):
    plt.scatter(x1[i],x2[i],color="black")
for i in range(7,14):
    plt.scatter(x1[i],x2[i],marker="*",color="red")
w1=2
w2=1
bias=0.5
learn=0.2
print("weight1\tweight2")
start = timeit.default_timer()
for i in range (0,1200):
    calc=([])
    y=([])
    res=([])
    calc2=([])
    for j in range(0,14):
        calc.append(w1*x1[j]+w2*x2[j]+bias)
        calc2.append(1/(1+math.exp(-calc[j])))
        res.append(calc2[j]-yd[j])
        if res[j]!=0:
            w1=w1+((-learn)*calc2[j]*(1-calc2[j])*res[j]*x1[j])
            w2=w2+((-learn)*calc2[j]*(1-calc2[j])*res[j]*x2[j])
            bias = bias + ((-learn) * calc2[j] * (1 - calc2[j]) * res[j])
            print('{:.4f}\t{:.4f}\t{:.4f}'.format(w1,w2,bias))
stop = timeit.default_timer()
print('Time: ', stop - start)
count=0
y2=([])
for i in range (0,14):
    if(calc2[i]==yd[i]):
        count=count+1
accuracy=(count/14)*100
error=100-accuracy
print('accuracy={:.4f}\terror={:.4f}'.format(accuracy,error))
for i in range (0,14):
    y2.append((-w1/w2)*x1[i]+(-bias/w2))
plt.plot(x1,y2)
plt.grid(True)
plt.show()