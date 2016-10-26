#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

vectors_data = np.loadtxt("Data11-17.txt")
print(vectors_data.shape)
print(len(vectors_data))



y = vectors_data[0:vectors_data.shape[0]//100,64]
y = np.reshape(y,(len(y),1))
print("y.shape")
print(y.shape)
y = np.reshape(y,-1)
print(y.shape)

a = vectors_data[0:vectors_data.shape[0],0:63]
b = vectors_data[0:vectors_data.shape[0],64:vectors_data.shape[1]]
c = np.hstack((a,b))

print(c.shape)

