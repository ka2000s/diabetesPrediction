#!/usr/bin/env python3
# Diabetes prediction
# Check the file globalVariables for adjustments
import sys
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import globalVariables as gl
from dataPreperation import readAndCleanDataSet, prepareTrainingData
from algorithms import calculateSupportVectorMachines, calculateStochasticGradientDescent, calculateNaiveBayes, \
    calculateDecisionTrees, calculateLogisticRegression, calculateKNearestNeighbors


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
names = []
# Stochastic Gradient Descent -> SGD
if True:
    name, accuracy, f1, specificity, sensitivity, scores = calculateStochasticGradientDescent(x_train, x_test, y_train, y_test)
    accuracies.append(scores)
    names.append(name)
    results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Support Vector Machines -> SVM
if True:
    name, accuracy, f1, specificity, sensitivity, scores = calculateSupportVectorMachines(x_train, x_test, y_train, y_test)
    accuracies.append(scores)
    names.append(name)
    results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Naive Bayes -> NB
if True:
    name, accuracy, f1, specificity, sensitivity, scores = calculateNaiveBayes(x_train, x_test, y_train, y_test)
    accuracies.append(scores)
    names.append(name)
    results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Decision Trees -> DT
if True:
    name, accuracy, f1, specificity, sensitivity, scores = calculateDecisionTrees(x_train, x_test, y_train, y_test)
    accuracies.append(scores)
    names.append(name)
    results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# Logistic Regression -> LR
if True:
    name, accuracy, f1, specificity, sensitivity, scores = calculateLogisticRegression(x_train, x_test, y_train, y_test)
    accuracies.append(scores)
    names.append(name)
    results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))
# K Nearest Neighbors -> KNN
if True:
    name, accuracy, f1, specificity, sensitivity, scores = calculateKNearestNeighbors(x_train, x_test, y_train, y_test)
    accuracies.append(scores)
    names.append(name)
    results.append(gl.Results(name, accuracy, f1, specificity, sensitivity))

if gl.savePlots:
    df = pd.DataFrame.from_dict(dict(zip(names, accuracies)))
    plt.figure(figsize=(35, 15), dpi=300)
    sns.boxplot(data=df)#, x="Models", y="Accuracy")
    filePathName = gl.plotFilePath + "model_comparision" + gl.fileFormat
    gl.savePlots(filePathName, plt)

if gl.printing:
    print("Model, Accuracy, Specificity, Sensitivity, F1")
    for i in results:
        print(i.name, ": ", i.accuracy, ", ", i.specificity, ", ", i.sensitivity, ", ", i.f1)
        
    if gl.latexOutput:
        print()
        print("% Latex Table")
        print("\\begin{table}[H]")
        print("\\renewcommand{\\arraystretch}{1.3}")
        print("\\begin{tabularx}{\\linewidth}{XXXXX}")
        print("Model & Accuracy & Specificity & Sensitivity & F1 \\\\")
        print("\\toprule")
        for i in results:
            print(i.name, " & ", format(i.accuracy,".6f"), " & ", format(i.specificity,".6f"), " & ", format(i.sensitivity,".6f"), " & ", format(i.f1,".6f"), " \\\\")
        print("\\end{tabularx}")
        print("\\caption{Model Results}")
        print("\\label{fig:results}")
        print("\\end{table}")
        

