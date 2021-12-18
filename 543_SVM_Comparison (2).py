#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.metrics.pairwise import laplacian_kernel


# In[2]:


#NOTE: download dataset at https://archive.ics.uci.edu/ml/datasets/banknote+authentication
#and replace filepath with that of your local machine
#THIS SHOULD BE THE ONLY CHANGE MADE IN ORDER TO RUN THE PROGRAM
with open(r"C:\Users\ethan\Downloads\data_banknote_authentication.txt") as file:
    csv_file = csv.reader(file, delimiter = ",")
    data = [[]]
    for line in csv_file:
        data.append(line) 
    data = data[1:]


# In[3]:


df = pd.DataFrame(data, columns =['variance', 'skewness', 'curtosis', 'entropy', 'Class'])
#put data into pandas dataframe
print(df)


# In[4]:


#Split up dataset into subgroups for creating validation and testing sets
realNotes = data[:762]
fakeNotes = data[-610:]
print(realNotes)
print('')
print(fakeNotes)
#print(z)


# In[5]:


#Create final validation set for kernel comparison with proportion of 
#real and fake banknotes corresponding to entire set - 10% of the dataset is
#set aside for this
validationData = []
for i in range(0, 61):
    validationData.append(fakeNotes[i])
for k in range(0, 76):
    validationData.append(realNotes[k])
print(len(validationData))
validationSet = pd.DataFrame(validationData, columns =['variance', 'skewness', 'curtosis', 'entropy', 'Class'])


# In[6]:


#Create ten training subsets for 10-fold cross validation process
#Tried to keep proportions relatively balanced for these as well
trainingSubsetsComplete = []
for z in range(0, 10):
    trainingSubsetTemp = []
    for i in range(0, 55):
        trainingSubsetTemp.append(fakeNotes[z*55 + 61 + i])
        if(z*55 + 61 + i == len(fakeNotes) - 1):
            break
    trainingSubsetsComplete.append(trainingSubsetTemp)
for z in range(0, 10):
    trainingSubsetTemp = []
    for i in range(0, 69):
        trainingSubsetTemp.append(realNotes[z*69 + 76 + i])
        if(z*69 + 76 + i == len(realNotes) - 1):
            break
    for k in range(0, len(trainingSubsetTemp)):
        trainingSubsetsComplete[z].append(trainingSubsetTemp[k])
trainingSubsets = []
for k in range(0, 10):
    trainingSubsets.append(pd.DataFrame(trainingSubsetsComplete[k], columns = ['variance', 'skewness', 'curtosis', 'entropy', 'Class']))
print(trainingSubsets)


# In[7]:


#Create pairs of training sets and testing sets for each fold in 10-fold cross validation
testZero = [trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
testZeroF = pd.concat(testZero)
testZeroPair = [testZeroF, trainingSubsets[0]]
testOne = [trainingSubsets[0], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
testOneF = pd.concat(testOne)
testOnePair = [testOneF, trainingSubsets[1]]
testTwo = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
testTwoF = pd.concat(testTwo)
testTwoPair = [testTwoF, trainingSubsets[2]]
testThree = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
testThreeF = pd.concat(testThree)
testThreePair = [testThreeF, trainingSubsets[3]]
testFour = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
testFourF = pd.concat(testFour)
testFourPair = [testFourF, trainingSubsets[4]]
testFive = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
testFiveF = pd.concat(testFive)
testFivePair = [testFiveF, trainingSubsets[5]]
testSix = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
testSixF = pd.concat(testSix)
testSixPair = [testSixF, trainingSubsets[6]]
testSeven = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[8], trainingSubsets[9]]
testSevenF = pd.concat(testSeven)
testSevenPair = [testSevenF, trainingSubsets[7]]
testEight = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[9]]
testEightF = pd.concat(testEight)
testEightPair = [testEightF, trainingSubsets[8]]
testNine = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8]]
testNineF = pd.concat(testNine)
testNinePair = [testNineF, trainingSubsets[9]]
fullTrain = [trainingSubsets[0], trainingSubsets[1], trainingSubsets[2], trainingSubsets[3], trainingSubsets[4], trainingSubsets[5], trainingSubsets[6], trainingSubsets[7], trainingSubsets[8], trainingSubsets[9]]
fullTrainingSet = pd.concat(fullTrain)
fullTrainingPair = [fullTrainingSet, validationSet]
#Initialize lists for runtime and accuracy comparisons
finalAccuracies = []
finalRuntimeTrain = []
finalRuntimeValidation = []


# In[8]:


#Train polynomial kernel svm using 10-fold cross validation - change kernel coefficient each iteration
x_col = ['variance', 'skewness', 'curtosis', 'entropy']
y_col = ['Class']
accuracies = []
kernelCoefficients = []
start = time.time()
for i in range(1, 20):
    kernelCoefficients.append(0.05 * i)
    totalCorrect = 0
    total = 1372
    svm = SVC(kernel='poly', degree = 3, gamma = 0.05 * i)
    svm.fit(testZeroPair[0][x_col].astype(float), np.ravel(testZeroPair[0][y_col]))
    p0 = svm.predict(testZeroPair[1][x_col].astype(float))
    svm.fit(testOnePair[0][x_col].astype(float), np.ravel(testOnePair[0][y_col]))
    p1 = svm.predict(testOnePair[1][x_col].astype(float))
    svm.fit(testTwoPair[0][x_col].astype(float), np.ravel(testTwoPair[0][y_col]))
    p2 = svm.predict(testTwoPair[1][x_col].astype(float))
    svm.fit(testThreePair[0][x_col].astype(float), np.ravel(testThreePair[0][y_col]))
    p3 = svm.predict(testThreePair[1][x_col].astype(float))
    svm.fit(testFourPair[0][x_col].astype(float), np.ravel(testFourPair[0][y_col]))
    p4 = svm.predict(testFourPair[1][x_col].astype(float))
    svm.fit(testFivePair[0][x_col].astype(float), np.ravel(testFivePair[0][y_col]))
    p5 = svm.predict(testFivePair[1][x_col].astype(float))
    svm.fit(testSixPair[0][x_col].astype(float), np.ravel(testSixPair[0][y_col]))
    p6 = svm.predict(testSixPair[1][x_col].astype(float))
    svm.fit(testSevenPair[0][x_col].astype(float), np.ravel(testSevenPair[0][y_col]))
    p7 = svm.predict(testSevenPair[1][x_col].astype(float))
    svm.fit(testEightPair[0][x_col].astype(float), np.ravel(testEightPair[0][y_col]))
    p8 = svm.predict(testEightPair[1][x_col].astype(float))
    svm.fit(testNinePair[0][x_col].astype(float), np.ravel(testNinePair[0][y_col]))
    p9 = svm.predict(testNinePair[1][x_col].astype(float))
    fullPredictSet = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    fullPredictSetF = []
    fullPredictSetF = np.concatenate(fullPredictSet)
    fullTrains = fullTrainingSet.to_numpy()
    for k in range(0, 1235):
         if(fullPredictSetF[k] == fullTrains[k][4]):
            totalCorrect += 1
    accuracies.append(totalCorrect/total)
end = time.time()
finalRuntimeTrain.append(end - start)
print(accuracies)
plt.plot(kernelCoefficients, accuracies)
plt.xlabel("Kernel Coefficient")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Kernel Coefficient for Polynomial Kernel SVM")
plt.show()


# In[9]:


#Hold kernel coefficient hyperparameter at optimal value and test on validation set
totalCorrect = 0
total = 137
start = time.time()
svm = SVC(kernel='poly', degree = 3, gamma = 0.45)
svm.fit(fullTrainingSet[x_col].astype(float), np.ravel(fullTrainingSet[y_col]))
predictions = svm.predict(validationSet[x_col].astype(float))
validationArr = validationSet.to_numpy()
for k in range(0, 137):
    if(predictions[k] == validationArr[k][4]):
        totalCorrect += 1
accuracy = totalCorrect/total
end = time.time()
print(accuracy)
finalRuntimeValidation.append(end - start)
finalAccuracies.append(accuracy)


# In[10]:


#Train sigmoid kernel svm using 10-fold cross validation - change kernel coefficient each iteration
x_col = ['variance', 'skewness', 'curtosis', 'entropy']
y_col = ['Class']
accuracies = []
kernelCoefficients = []
start = time.time()
for i in range(1, 20):
    kernelCoefficients.append(0.05 * i)
    totalCorrect = 0
    total = 1372
    svm = SVC(kernel='sigmoid', gamma = 0.05 * i)
    svm.fit(testZeroPair[0][x_col].astype(float), np.ravel(testZeroPair[0][y_col]))
    p0 = svm.predict(testZeroPair[1][x_col].astype(float))
    svm.fit(testOnePair[0][x_col].astype(float), np.ravel(testOnePair[0][y_col]))
    p1 = svm.predict(testOnePair[1][x_col].astype(float))
    svm.fit(testTwoPair[0][x_col].astype(float), np.ravel(testTwoPair[0][y_col]))
    p2 = svm.predict(testTwoPair[1][x_col].astype(float))
    svm.fit(testThreePair[0][x_col].astype(float), np.ravel(testThreePair[0][y_col]))
    p3 = svm.predict(testThreePair[1][x_col].astype(float))
    svm.fit(testFourPair[0][x_col].astype(float), np.ravel(testFourPair[0][y_col]))
    p4 = svm.predict(testFourPair[1][x_col].astype(float))
    svm.fit(testFivePair[0][x_col].astype(float), np.ravel(testFivePair[0][y_col]))
    p5 = svm.predict(testFivePair[1][x_col].astype(float))
    svm.fit(testSixPair[0][x_col].astype(float), np.ravel(testSixPair[0][y_col]))
    p6 = svm.predict(testSixPair[1][x_col].astype(float))
    svm.fit(testSevenPair[0][x_col].astype(float), np.ravel(testSevenPair[0][y_col]))
    p7 = svm.predict(testSevenPair[1][x_col].astype(float))
    svm.fit(testEightPair[0][x_col].astype(float), np.ravel(testEightPair[0][y_col]))
    p8 = svm.predict(testEightPair[1][x_col].astype(float))
    svm.fit(testNinePair[0][x_col].astype(float), np.ravel(testNinePair[0][y_col]))
    p9 = svm.predict(testNinePair[1][x_col].astype(float))
    fullPredictSet = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    fullPredictSetF = []
    fullPredictSetF = np.concatenate(fullPredictSet)
    fullTrains = fullTrainingSet.to_numpy()
    for k in range(0, 1235):
         if(fullPredictSetF[k] == fullTrains[k][4]):
            totalCorrect += 1
    accuracies.append(totalCorrect/total)
end = time.time()
finalRuntimeTrain.append(end - start)
print(accuracies)
plt.plot(kernelCoefficients, accuracies)
plt.xlabel("Kernel Coefficient")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Kernel Coefficient for Sigmoid Kernel SVM")
plt.show()


# In[11]:


#Hold kernel coefficient hyperparameter at optimal value and test on validation set
totalCorrect = 0
total = 137
start = time.time()
svm = SVC(kernel='sigmoid', gamma = 0.55)
svm.fit(fullTrainingSet[x_col].astype(float), np.ravel(fullTrainingSet[y_col]))
predictions = svm.predict(validationSet[x_col].astype(float))
validationArr = validationSet.to_numpy()
for k in range(0, 137):
    if(predictions[k] == validationArr[k][4]):
        totalCorrect += 1
accuracy = totalCorrect/total
end = time.time()
print(accuracy)
finalRuntimeValidation.append(end - start)
finalAccuracies.append(accuracy)


# In[12]:


#Test linear kernel svm on dataset - no applicable hyperparameters to tune so only train on full 
#training set and test on final validation
totalCorrect = 0
total = 137
start = time.time()
svm = SVC(kernel='linear')
svm.fit(fullTrainingSet[x_col].astype(float), np.ravel(fullTrainingSet[y_col]))
predictions = svm.predict(validationSet[x_col].astype(float))
validationArr = validationSet.to_numpy()
for k in range(0, 137):
    if(predictions[k] == validationArr[k][4]):
        totalCorrect += 1
accuracy = totalCorrect/total
end = time.time()
print(accuracy)
finalRuntimeValidation.append(end - start)
finalAccuracies.append(accuracy)


# In[13]:


#Train rbf kernel svm using 10-fold cross validation - change kernel coefficient each iteration
x_col = ['variance', 'skewness', 'curtosis', 'entropy']
y_col = ['Class']
accuracies = []
kernelCoefficients = []
start = time.time()
for i in range(1, 20):
    kernelCoefficients.append(0.05 * i)
    totalCorrect = 0
    total = 1372
    svm = SVC(kernel='rbf', gamma = 0.05 * i)
    svm.fit(testZeroPair[0][x_col].astype(float), np.ravel(testZeroPair[0][y_col]))
    p0 = svm.predict(testZeroPair[1][x_col].astype(float))
    svm.fit(testOnePair[0][x_col].astype(float), np.ravel(testOnePair[0][y_col]))
    p1 = svm.predict(testOnePair[1][x_col].astype(float))
    svm.fit(testTwoPair[0][x_col].astype(float), np.ravel(testTwoPair[0][y_col]))
    p2 = svm.predict(testTwoPair[1][x_col].astype(float))
    svm.fit(testThreePair[0][x_col].astype(float), np.ravel(testThreePair[0][y_col]))
    p3 = svm.predict(testThreePair[1][x_col].astype(float))
    svm.fit(testFourPair[0][x_col].astype(float), np.ravel(testFourPair[0][y_col]))
    p4 = svm.predict(testFourPair[1][x_col].astype(float))
    svm.fit(testFivePair[0][x_col].astype(float), np.ravel(testFivePair[0][y_col]))
    p5 = svm.predict(testFivePair[1][x_col].astype(float))
    svm.fit(testSixPair[0][x_col].astype(float), np.ravel(testSixPair[0][y_col]))
    p6 = svm.predict(testSixPair[1][x_col].astype(float))
    svm.fit(testSevenPair[0][x_col].astype(float), np.ravel(testSevenPair[0][y_col]))
    p7 = svm.predict(testSevenPair[1][x_col].astype(float))
    svm.fit(testEightPair[0][x_col].astype(float), np.ravel(testEightPair[0][y_col]))
    p8 = svm.predict(testEightPair[1][x_col].astype(float))
    svm.fit(testNinePair[0][x_col].astype(float), np.ravel(testNinePair[0][y_col]))
    p9 = svm.predict(testNinePair[1][x_col].astype(float))
    fullPredictSet = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    fullPredictSetF = []
    fullPredictSetF = np.concatenate(fullPredictSet)
    fullTrains = fullTrainingSet.to_numpy()
    for k in range(0, 1235):
         if(fullPredictSetF[k] == fullTrains[k][4]):
            totalCorrect += 1
    accuracies.append(totalCorrect/total)
end = time.time()
finalRuntimeTrain.append(end - start)
print(accuracies)
plt.plot(kernelCoefficients, accuracies)
plt.xlabel("Kernel Coefficient")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Kernel Coefficient for RBF Kernel SVM")
plt.show()


# In[14]:


#Hold kernel coefficient hyperparameter at optimal value and test on validation set
totalCorrect = 0
total = 137
start = time.time()
svm = SVC(kernel='rbf', gamma = 0.45)
svm.fit(fullTrainingSet[x_col].astype(float), np.ravel(fullTrainingSet[y_col]))
predictions = svm.predict(validationSet[x_col].astype(float))
validationArr = validationSet.to_numpy()
for k in range(0, 137):
    if(predictions[k] == validationArr[k][4]):
        totalCorrect += 1
accuracy = totalCorrect/total
end = time.time()
finalRuntimeValidation.append(end - start)
print(accuracy)
finalAccuracies.append(accuracy)


# In[15]:


#Train laplacian kernel svm using 10-fold cross validation - change kernel coefficient each iteration
#NOTE: I left the training process in, but no change was observed in performance for this kernel when
#changing kernel coefficient - gamma. I also tested a much larger swath of values and still observed no change.
x_col = ['variance', 'skewness', 'curtosis', 'entropy']
y_col = ['Class']
accuracies = []
kernelCoefficients = []
start = time.time()
for i in range(1, 20):
    kernelCoefficients.append(0.05 * i)
    totalCorrect = 0
    total = 1372
    svm = SVC(kernel=laplacian_kernel, gamma = 0.05 * i)
    svm.fit(testZeroPair[0][x_col].astype(float), np.ravel(testZeroPair[0][y_col]))
    p0 = svm.predict(testZeroPair[1][x_col].astype(float))
    svm.fit(testOnePair[0][x_col].astype(float), np.ravel(testOnePair[0][y_col]))
    p1 = svm.predict(testOnePair[1][x_col].astype(float))
    svm.fit(testTwoPair[0][x_col].astype(float), np.ravel(testTwoPair[0][y_col]))
    p2 = svm.predict(testTwoPair[1][x_col].astype(float))
    svm.fit(testThreePair[0][x_col].astype(float), np.ravel(testThreePair[0][y_col]))
    p3 = svm.predict(testThreePair[1][x_col].astype(float))
    svm.fit(testFourPair[0][x_col].astype(float), np.ravel(testFourPair[0][y_col]))
    p4 = svm.predict(testFourPair[1][x_col].astype(float))
    svm.fit(testFivePair[0][x_col].astype(float), np.ravel(testFivePair[0][y_col]))
    p5 = svm.predict(testFivePair[1][x_col].astype(float))
    svm.fit(testSixPair[0][x_col].astype(float), np.ravel(testSixPair[0][y_col]))
    p6 = svm.predict(testSixPair[1][x_col].astype(float))
    svm.fit(testSevenPair[0][x_col].astype(float), np.ravel(testSevenPair[0][y_col]))
    p7 = svm.predict(testSevenPair[1][x_col].astype(float))
    svm.fit(testEightPair[0][x_col].astype(float), np.ravel(testEightPair[0][y_col]))
    p8 = svm.predict(testEightPair[1][x_col].astype(float))
    svm.fit(testNinePair[0][x_col].astype(float), np.ravel(testNinePair[0][y_col]))
    p9 = svm.predict(testNinePair[1][x_col].astype(float))
    fullPredictSet = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    fullPredictSetF = []
    fullPredictSetF = np.concatenate(fullPredictSet)
    fullTrains = fullTrainingSet.to_numpy()
    for k in range(0, 1235):
         if(fullPredictSetF[k] == fullTrains[k][4]):
            totalCorrect += 1
    accuracies.append(totalCorrect/total)
end = time.time()
finalRuntimeTrain.append(end - start)
print(accuracies)
plt.plot(kernelCoefficients, accuracies)
plt.xlabel("Kernel Coefficient")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Kernel Coefficient for Laplacian Kernel SVM")
plt.show()


# In[16]:


#Test laplacian kernel svm on final validation set
totalCorrect = 0
total = 137
start = time.time()
svm = SVC(kernel=laplacian_kernel)
svm.fit(fullTrainingSet[x_col].astype(float), np.ravel(fullTrainingSet[y_col]))
predictions = svm.predict(validationSet[x_col].astype(float))
validationArr = validationSet.to_numpy()
for k in range(0, 137):
    if(predictions[k] == validationArr[k][4]):
        totalCorrect += 1
accuracy = totalCorrect/total
end = time.time()
finalRuntimeValidation.append(end - start)
print(accuracy)
finalAccuracies.append(accuracy)


# In[17]:


#Show plot for accuracy of each kernel on final validation set
print(finalAccuracies)
plt.scatter(['1', '2', '3', '4', '5'], finalAccuracies)
plt.xlabel("Kernel Number")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Kernel")
plt.show()
#Show plot of runtime for each kernel during training - note linear not included for this, as no 
#hyperparameters were tuned. Linear kernel was included solely for sake of comparison.
plt.scatter(['1', '2', '3', '4'], finalRuntimeTrain)
plt.xlabel("Kernel Number")
plt.ylabel("Runtime during 10-fold Cross Validation (s)")
plt.title("Runtime during 10-fold Cross Validation vs. Kernel")
plt.show()
#Show plot of runtime for each kernel during testing on final validation set
plt.scatter(['1', '2', '3', '4', '5'], finalRuntimeValidation)
plt.xlabel("Kernel Number")
plt.ylabel("Runtime during Testing on Validation Set (s)")
plt.title("Runtime during Testing on Validation Set vs. Kernel")
plt.show()

