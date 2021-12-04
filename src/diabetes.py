#!/usr/bin/env python3
# Project for Technology and Diabetes Management
# Diabetes prediction
# Group 1: Daniel Bürgler & Daniel Monti & Camille Serquet
# Check the file globalVariables for adjustments
import sys
import globalVariables
from dataPreperation import readAndCleanDataSet, prepareTrainingData
from algorithms import calculateSupportVectorMachines, calculateStochasticGradientDescent, calculateNaiveBayes, \
    calculateDecisionTrees, calculateLogisticRegression, calculateKNearestNeighbors


# Redirect stdout to file when global variable printToFile is set
if globalVariables.printToFile:
    logfile = open(globalVariables.logFilePath, 'w')
    sys.stdout = logfile

# Get the Clean Dataset
data = readAndCleanDataSet()
# Prepare Dataset for training
[x_train, x_test, y_train, y_test] = prepareTrainingData(data)

# Calculate all 6 Algorithms
results = []
# Stochastic Gradient Descent -> SGD
name, accuracy, f1, specificity, sensitivity = calculateStochasticGradientDescent(x_train, x_test, y_train, y_test)
results.append(globalVariables.Results(name, accuracy, f1, specificity, sensitivity))
# Support Vector Machines -> SVM
name, accuracy, f1, specificity, sensitivity = calculateSupportVectorMachines(x_train, x_test, y_train, y_test)
results.append(globalVariables.Results(name, accuracy, f1, specificity, sensitivity))
# Naive Bayes -> NB
name, accuracy, f1, specificity, sensitivity = calculateNaiveBayes(x_train, x_test, y_train, y_test)
results.append(globalVariables.Results(name, accuracy, f1, specificity, sensitivity))
# Decision Trees -> DT
name, accuracy, f1, specificity, sensitivity = calculateDecisionTrees(x_train, x_test, y_train, y_test)
results.append(globalVariables.Results(name, accuracy, f1, specificity, sensitivity))
# Logistic Regression -> LR
name, accuracy, f1, specificity, sensitivity = calculateLogisticRegression(x_train, x_test, y_train, y_test)
results.append(globalVariables.Results(name, accuracy, f1, specificity, sensitivity))
# K Nearest Neighbors -> KNN
name, accuracy, f1, specificity, sensitivity = calculateKNearestNeighbors(x_train, x_test, y_train, y_test)
results.append(globalVariables.Results(name, accuracy, f1, specificity, sensitivity))

if globalVariables.printing:
    print("Model, Accuracy, Specificity, Sensitivity, F1")
    for i in results:
        print(i.name, ": ", i.accuracy, ", ", i.specificity, ", ", i.sensitivity, ", ", i.f1)
