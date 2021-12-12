#!/usr/bin/env python3
# Diabetes prediction
# Check the file globalVariables for adjustments
import sys
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import globalVariables as gl
from dataPreperation import readAndCleanDataSet, prepareTrainingData
from algorithms import calculateSupportVectorMachines, calculateStochasticGradientDescent, \
    calculateNaiveBayes, calculateDecisionTrees, calculateLogisticRegression, \
    calculateKNearestNeighbors

# Redirect stdout to file when global variable printToFile is set
if gl.printToFile:
    logfile = open(gl.logFilePath, 'w')
    sys.stdout = logfile

# Get the Clean Dataset
data = readAndCleanDataSet()
# Prepare Dataset for training
[x_train, x_test, y_train, y_test] = prepareTrainingData(data)

# Calculate all 6 Algorithms
results = []
accuracies = []
f1s = []
names = []
# Stochastic Gradient Descent -> SGD
name, accuracy, f1, specificity, sensitivity, accuracyScores, f1Scores = \
    calculateStochasticGradientDescent(x_train, x_test, y_train, y_test)
accuracies.append(accuracyScores)
f1s.append(f1Scores)
names.append(name)
results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Support Vector Machines -> SVM
name, accuracy, f1, specificity, sensitivity, accuracyScores, f1Scores = \
    calculateSupportVectorMachines(x_train, x_test, y_train, y_test)
accuracies.append(accuracyScores)
f1s.append(f1Scores)
names.append(name)
results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Naive Bayes -> NB
name, accuracy, f1, specificity, sensitivity, accuracyScores, f1Scores = \
    calculateNaiveBayes(x_train, x_test, y_train, y_test)
accuracies.append(accuracyScores)
f1s.append(f1Scores)
names.append(name)
results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Decision Trees -> DT
name, accuracy, f1, specificity, sensitivity, accuracyScores, f1Scores = \
    calculateDecisionTrees(x_train, x_test, y_train, y_test)
accuracies.append(accuracyScores)
f1s.append(f1Scores)
names.append(name)
results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Logistic Regression -> LR
name, accuracy, f1, specificity, sensitivity, accuracyScores, f1Scores = \
    calculateLogisticRegression(x_train, x_test, y_train, y_test)
accuracies.append(accuracyScores)
f1s.append(f1Scores)
names.append(name)
results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# K Nearest Neighbors -> KNN
name, accuracy, f1, specificity, sensitivity, accuracyScores, f1Scores = \
    calculateKNearestNeighbors(x_train, x_test, y_train, y_test)
accuracies.append(accuracyScores)
f1s.append(f1Scores)
names.append(name)
results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))

print("Model, Accuracy, Specificity, Sensitivity, F1")
for i in results:
    print(i.name, ": ", i.accuracy, ", ", i.specificity, ", ", i.sensitivity, ", ", i.f1)

if gl.savePlots:
    df = pd.DataFrame.from_dict(dict(zip(names, accuracies)))
    f, axes = plt.subplots(2, 1, figsize=(30, 25))
    sns.set(font_scale=2)
    axes[0].set_title('Accuracy', fontsize=30)
    axes[1].set_title('F1 Score', fontsize=30)
    sns.boxplot(data=df, orient='v', ax=axes[0])
    df = pd.DataFrame.from_dict(dict(zip(names, f1s)))
    sns.boxplot(data=df, orient='v', ax=axes[1])
    filePathName = gl.plotFilePath + "model_comparision" + gl.fileFormat
    gl.savePlots(filePathName, plt)
    
