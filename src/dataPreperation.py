#!/usr/bin/env python3
import numpy as np
import pandas as pd
import globalVariables as gl
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set(color_codes=True)

def readAndCleanDataSet():
    data = pd.read_csv(gl.filePath, delimiter=gl.fileDelimiter)

    # Exchange zeros with Nan
    dataClean = data.copy(deep=True)
    dataClean[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']] = dataClean[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']].replace(0, np.NaN)
        
    if gl.printing:
        print(data.head(5))
        print(data.info())
        print(dataClean.isnull().sum())
        
    # Replace NaN values with their distribution
    dataClean['Glucose'].fillna(dataClean['Glucose'].mean(), inplace=True)
    dataClean['BloodPressure'].fillna(dataClean['BloodPressure'].mean(), inplace=True)
    dataClean['SkinThickness'].fillna(dataClean['SkinThickness'].median(), inplace=True)
    dataClean['Insulin'].fillna(dataClean['Insulin'].median(), inplace=True)
    dataClean['BMI'].fillna(dataClean['BMI'].median(), inplace=True)

    if gl.savePlots:
        # Outcome
        plt.figure(figsize=(8, 8))
        pie = data['Outcome'].value_counts()
        colors = ['moccasin', 'coral']
        labels = ['0 - Non Diabetic', '1 - Diabetic']
        sns.set(font_scale=1.5)
        plt.pie(pie, autopct="%.2f%%", colors=colors)
        plt.legend(labels, loc='lower left')

        filePathName = gl.plotFilePath + "dataset_diabetes_and_non_diabetes" + gl.fileFormat
        gl.savePlots(filePathName, plt)
        
        plt.figure(figsize=(7, 5))
        sns.set_theme(style="darkgrid")
        sns.countplot(x= "Outcome", data = data).set(xticklabels=["Diabetes", "No Diabetes"])
        plt.xlabel("Diabetes vs No Diabetes Comparison")
        plt.ylabel("Number of People")
        filePathName = gl.plotFilePath + "dataset_diabetes_and_non_diabetes_2" + gl.fileFormat
        gl.savePlots(filePathName, plt)

        # Distribution and BoxPlot
        fig, ax = plt.subplots(8,2, figsize=(25,25))
        fig.tight_layout()
        sns.set(font_scale = 1.5)
        sns.distplot(data.Pregnancies, ax = ax[0,0], color = 'orange')
        sns.distplot(data.Glucose, ax = ax[1,0], color = 'red')
        sns.distplot(data.BloodPressure, ax = ax[2,0], color = 'seagreen')
        sns.distplot(data.SkinThickness, ax = ax[3,0], color = 'purple')
        sns.distplot(data.Insulin, ax = ax[4,0], color = 'deeppink')
        sns.distplot(data.BMI, ax = ax[5,0], color = 'brown')
        sns.distplot(data.DiabetesPedigreeFunction, ax = ax[6,0], color = 'royalblue')
        sns.distplot(data.Age, ax = ax[7,0], color = 'coral')
        sns.boxplot(data.Pregnancies, ax = ax[0,1], color = 'orange')
        sns.boxplot(data.Glucose, ax = ax[1,1], color = 'red')
        sns.boxplot(data.BloodPressure, ax = ax[2,1], color = 'seagreen')
        sns.boxplot(data.SkinThickness, ax = ax[3,1], color = 'purple')
        sns.boxplot(data.Insulin, ax = ax[4,1], color = 'deeppink')
        sns.boxplot(data.BMI, ax = ax[5,1], color = 'brown')
        sns.boxplot(data.DiabetesPedigreeFunction, ax = ax[6,1], color = 'royalblue')
        sns.boxplot(data.Age, ax = ax[7,1], color = 'coral')
        filePathName = gl.plotFilePath + "dataset_distibutionAndBoxPlot" + gl.fileFormat
        gl.savePlots(filePathName, plt)

        # Data Histogram
        data.hist(figsize = (11,11), color="#008080")
        filePathName = gl.plotFilePath + "dataset_histogram" + gl.fileFormat
        gl.savePlots(filePathName, plt)


        # Correlation Plot
        correlation = dataClean.corr()
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1.5)
        sns.heatmap(correlation, annot=True, cmap='plasma', vmin=-1, vmax=1, linecolor='white', linewidths=1)
        sns.set(font_scale=1.5)
        sns.pairplot(data=dataClean, hue='Outcome', diag_kind='kde', palette='Set2')
        filePathName = gl.plotFilePath + "dataset_correlation" + gl.fileFormat
        gl.savePlots(filePathName, plt)

    return dataClean


def prepareTrainingData(data):
    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return train_test_split(x, y, test_size=gl.testSize, shuffle=True, random_state=gl.randomState)
