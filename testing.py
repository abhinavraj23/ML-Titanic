import csv
import numpy as np
from sigmoid import sigmoid 
from decimal import Decimal
from costFunction import costFunction
import matplotlib.pyplot as plt
from time import time 

rawData = []
data = []
result = []
pClass = []
sex = []
age = []
sibSp = []
parch = []
fare = []
newData = []
newTheta = []
finalData = []
survived = []
# read the csv file
i = 0
with open('test.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        rawData.append(row)
        if i == 0:
        	i = i+1
        else:	
         	sex.append(int(row[3] == "male"))
csvFile.close()        

# Data extraction from the csv file 
data = np.array(rawData)
ids = data[1:,0]
pClass = data[1:,1]
age = data[1:,4]
sibSp = data[1:,5]
parch = data[1:,6]
fare = data[1:,8]


theta = [-0.00182446,-0.07301107,-0.00019174,-0.0005875,0.0767088,-0.00236963]
theta = np.array(theta)

i=0
while i < len(ids):
    newData.append([float(pClass[i] or 0),float(age[i] or 0),float(sibSp[i] or 0),float(parch[i] or 0),float(fare[i] or 0),int(sex[i] or 0)])
    i+=1
m = len(ids)
survived = np.zeros([m,1])
newData = np.array(newData)
for j in range(m):
	arr = np.array([newData[j]])
	if(sigmoid(theta,arr) >= 0.5):
		survived[j] = 1
	else:
		survived[j] = 0;

i = 0
result = np.zeros([m,1])
with open('gender_submission.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if i == 0:
        	i = i+1
        else:	
         	result[i-1] = (int(row[1]))
         	i+=1
csvFile.close()
result = np.array(result)

print(len(survived))
correct = 0;
for z in range(len(survived)):
	if(survived[z] == [result[z]]):
		correct = correct + 1
accuracy = float(float(correct)/float(len(survived)))*100
print(accuracy)
# print(survived)
