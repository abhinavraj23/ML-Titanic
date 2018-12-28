import numpy as np
from sigmoid import sigmoid 

def costFunction(theta,data,result):
	m = len(data)
	result = np.array(result)
	data = np.array(data)
	theta = np.array(theta)
	value = np.dot(result.transpose(),np.log(sigmoid(theta,data)).transpose()) - np.dot((1 - result).transpose(),np.log(1 - sigmoid(theta,data)).transpose())
	return -value/m