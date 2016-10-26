print __doc__

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


from sklearn.metrics import r2_score

Data = np.loadtxt('result.txt')
Data = Data.reshape(40,60)
print(Data.shape)

C = np.zeros((2,Data.shape[1]),dtype = np.float)

from sklearn.linear_model import Lasso

alpha = 120
lasso = Lasso(alpha=alpha)

for i in range(Data.shape[1]-2):
  res = i+2
  X_train = Data[:,0:res]
  Y_ = Data[:,res]
  Y_pred_lasso = lasso.fit(X_train, Y_).predict(X_train)
  r2_score_lasso = r2_score(Y_, Y_pred_lasso)

  a = lasso.coef_.reshape(1,res)

  temp = np.zeros((1,Data.shape[1]-res),dtype = np.float)
  lasso.coef_expand = np.hstack((a,temp))
  C = np.vstack((C,lasso.coef_expand))
    
    
#print(C)

A = np.zeros(C.shape[1],dtype = np.float)
print("A.shape")
print(A.shape)
for i in range(C.shape[1]):
  for j in range(C.shape[0]):
    A[i] = A[i] + abs(C[j][i])

A = A/(C.shape[0]) 

plt.figure(figsize=(8,4))
plt.plot(A,label="A",color="red",linewidth=2)
plt.xlabel("Time(s)")
plt.ylabel("A")
plt.title("PyPlot First Example")
plt.legend()
plt.show()
print(A)   
 
    
