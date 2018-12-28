import numpy as np

def sigmoid(theta,features) :
	z = np.dot(theta.transpose(),features.transpose())
	return (1./(1+np.exp(-z)))