#!/usr/bin/python

import numpy as np

import collections
import os

import tensorflow as tf
import matplotlib.pyplot as plt

def read_data(filename):

	read_data = np.loadtxt(filename)

	#a = read_data[0:100,64]
	#b = read_data[100:read_data.shape[0],64]
	#labels_data = np.hstack((a,b))

	labels_data = read_data[0:read_data.shape[0],64]
	
	for i in range(len(labels_data)):
		if (labels_data[i] > 40):
			labels_data[i] = 2
		else:
			labels_data[i] = 0

	for i in range(len(labels_data)-1):
		if (labels_data[i] == 0 and labels_data[i+1] == 1):
			labels_data[i-20:i] = np.ones(20)


	a = read_data[0:read_data.shape[0],18:25]
	b = read_data[0:read_data.shape[0],37:55]
	a = np.hstack((a,b))
	b = read_data[0:read_data.shape[0],70:73]
	a = np.hstack((a,b))
	b = read_data[0:read_data.shape[0],74:87]
	a = np.hstack((a,b))
	
	vectors_data = a

	print(vectors_data.shape)

	return vectors_data,labels_data


def ptb_iterator(vectors_data, labels_data, batch_size, num_steps):
  data_len = len(vectors_data)
  batch_len = data_len // batch_size
  v_data = np.zeros([batch_size, batch_len,vectors_data.shape[1]], dtype = np.float)
  l_data = np.zeros([batch_size, batch_len], dtype = np.int32)
  for i in range(batch_size):
    v_data[i] = vectors_data[batch_len * i:batch_len * (i + 1),]
    l_data[i] = labels_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = v_data[:, i*num_steps:(i+1)*num_steps,:]
    y = l_data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)


