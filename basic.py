import csv
import numpy as np
from sigmoid import sigmoid 
from decimal import Decimal
from costFunction import costFunction
import matplotlib.pyplot as plt
from time import time 

rawData = []
data = []
survived = []
result = []
pClass = []
sex = []
age = []
sibSp = []
parch = []
fare = []
newData = []
newTheta = []
# read the csv file
i = 0
with open('train.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        rawData.append(row)
        if i == 0:
        	i = i+1
        else:	
         	sex.append(int(row[4] == "male"))

csvFile.close()        

# Data extraction from the csv file 
data = np.array(rawData)
survived = data[1:,1]
pClass = data[1:,2]
age = data[1:,5]
sibSp = data[1:,6]
parch = data[1:,7]
fare = data[1:,9]

# Initalised all the features as zeros
theta = np.zeros([1,6])
newTheta = np.zeros([6,1])
theta = np.array(theta)
theta = theta.transpose()
m =len(data)

# New data Generated 
i=0
while i < len(survived):
    newData.append([float(pClass[i] or 0),float(age[i] or 0),float(sibSp[i] or 0),float(parch[i] or 0),float(fare[i] or 0),int(sex[i] or 0)])
    result.append(int(survived[i] or 0))
    i+=1

newData = np.array(newData)
varData = newData.transpose()

#print(arr)

result = np.array([result])
result = result.transpose()

X = range(len(result))


#Gradient Descent algorithm
previousTheta = theta
alpha = 0.000005
m = len(newData)
while(costFunction(previousTheta,newData,result) >= costFunction(theta,newData,result)):
    Y = sigmoid(theta,newData)
    plt.scatter(X,Y)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    previousTheta = theta
    for i in range(6):
        arr = np.array([newData[i]])
        arr2 = np.array([varData[i]])
        newTheta[i] = theta[i] - (alpha/m)*np.dot((sigmoid(theta,arr) - result).transpose(),arr2.transpose())
    theta = [newTheta[0],newTheta[1],newTheta[2],newTheta[3],newTheta[4],newTheta[5]]
    theta = np.array(theta)
    print(previousTheta,theta)
    # Y = sigmoid(previousTheta,newData)
# plt.scatter(X,Y)
# plt.show()

Y = sigmoid(theta,newData)
plt.scatter(X,Y)
plt.show()

print(theta)

