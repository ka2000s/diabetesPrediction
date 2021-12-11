#!/usr/bin/env python3
import time
import globalVariables as gl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns  # statistical plot
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier  # K Nearest Neighbors
from sklearn.svm import SVC  # Support Vector Machines
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')


def calculateStochasticGradientDescent(x_train, x_test, y_train, y_test):
    # Test Parameters to evaluate
    testParameters = [{
        "loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 
        "penalty": ['l2', 'l1', 'elasticnet'], "class_weight": ['balanced'], "l1_ratio": np.linspace(0, 1, num=10), "alpha": np.power(10, np.arange(-4, 1, dtype=float))         
    }]
    best_parameters = []
    best_score = []
    # Scaling the train sets 
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train) 
    x_test_scaled = scaler.transform(x_test) 
    if gl.printing and gl.verboselevel > 1:
        print("Stochastic Gradient Descent GridSeacrh on scaled an unscaled dataset")
    for trainset in [x_train_scaled, x_train]:
        sgd = SGDClassifier(random_state=True)
        sgd_cv = GridSearchCV(sgd, testParameters, cv=gl.cv, verbose=gl.verboselevel, n_jobs=-1, scoring='accuracy')
        sgd_cv.fit(trainset, y_train)
                             
        best_parameters.append(sgd_cv.best_params_)
        best_score.append(sgd_cv.best_score_)
        
                             
    if best_score[0] >= best_score[1]:
        # Scaled Dataset
        sgd = SGDClassifier(**best_parameters[0])
        sgd.fit(x_train_scaled, y_train)
        y_predict = sgd.predict(x_test_scaled)
        print("Scaled Model has better accuracy of: ", best_score[0], " vs: ", best_score[1])
        print("Best Parameters are: ", best_parameters[0])
        scores = cross_val_score(estimator=sgd,
                         X=x_test_scaled, y=y_test,
                         cv=gl.cv, scoring='accuracy')
    else:
        # Non Scaled Dataset
        sgd = SGDClassifier(**best_parameters[1])
        sgd.fit(x_train, y_train)
        y_predict = sgd.predict(x_test)
        print("Non scaled Model has better accuracy of: ", best_score[1], " vs: ", best_score[0])
        print("Best Parameters are: ", best_parameters[1])
        scores = cross_val_score(estimator=sgd,
                         X=x_test, y=y_test,
                         cv=gl.cv, scoring='accuracy')

    return getCompareParameters("Stochastic Gradient Descent", y_test, y_predict, scores)


def calculateSupportVectorMachines(x_train, x_test, y_train, y_test):
    # Test Parameters to evaluate
    testParameters = [
        {"kernel": ["linear"], "C": [0.9, 1, 2, 10], "class_weight": ['balanced']},
        {"kernel": ["rbf"], "gamma": ['scale', 'auto'], "C": [0.9, 1, 2, 10], "class_weight": ['balanced']},
        {"kernel": ["poly"], "gamma": ['scale'], "C": [0.9, 1, 2, 10], "degree": [2, 3, 4, 5, 6, 7 ,8] , "coef0": [0.0], "class_weight": ['balanced']},
        {"kernel": ["sigmoid"], "gamma": ['scale', 'auto'], "C": [0.9, 1, 2, 10], "coef0": [0.0], "class_weight": ['balanced']},
    ]
    best_parameters = []
    best_score = []
    # Scaling the train sets 
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train) 
    x_test_scaled = scaler.transform(x_test) 
    if gl.printing  and gl.verboselevel > 1:
        print("Support Vector Machines GridSeacrh on scaled an unscaled dataset")
    for trainset in [x_train_scaled, x_train]:
        # Set the parameters by cross-validation
        svm = SVC(cache_size=5000, probability=False)
        svm_cv = GridSearchCV(svm, testParameters, scoring='accuracy', refit=False, cv=gl.cv, verbose=gl.verboselevel, n_jobs=-1) # TODO verbose level
        svm_cv.fit(trainset, y_train)
            
        best_parameters.append(svm_cv.best_params_)
        best_score.append(svm_cv.best_score_)
        
        
    if best_score[0] >= best_score[1]:
        # Scaled Dataset
        svm = SVC(**best_parameters[0])
        svm.fit(x_train_scaled, y_train)
        y_predict = svm.predict(x_test_scaled)
        print("Scaled Model has better accuracy of: ", best_score[0], " vs: ", best_score[1])
        print("Best Parameters are: ", best_parameters[0])
        scores = cross_val_score(estimator=svm,
                         X=x_test_scaled, y=y_test,
                         cv=gl.cv, scoring='accuracy')
    else:
        # Non Scaled Dataset
        svm = SVC(**best_parameters[1])
        svm.fit(x_train, y_train)
        y_predict = svm.predict(x_test)
        print("Non scaled Model has better accuracy of: ", best_score[1], " vs: ", best_score[0])
        print("Best Parameters are: ", best_parameters[1])
        scores = cross_val_score(estimator=svm,
                         X=x_test, y=y_test,
                         cv=gl.cv, scoring='accuracy')
   
    return getCompareParameters("Support Vector Machines", y_test, y_predict, scores)


def calculateNaiveBayes(x_train, x_test, y_train, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    scores = cross_val_score(estimator=model,
                         X=x_test, y=y_test,
                         cv=gl.cv, scoring='accuracy') 

    return getCompareParameters("Naive Bayes", y_test, y_predict, scores)


def calculateDecisionTrees(x_train, x_test, y_train, y_test):
    # Test Parameters to evaluate
    testParameters = [
        {"criterion": ['gini', 'entropy'], "splitter": ['best', 'random']}
    ]
    dt = DecisionTreeClassifier()
    dt_cv = GridSearchCV(dt, testParameters, scoring='accuracy', cv=gl.cv, verbose=gl.verboselevel, n_jobs=-1)
    dt_cv.fit(x_train, y_train)
    # Use the best parameters
    print("Best Parameters are: ", dt_cv.best_params_)
    dt = DecisionTreeClassifier(**dt_cv.best_params_)
    dt.fit(x_train, y_train)
    y_predict = dt.predict(x_test) # estimated targets as returned by the classifier
    scores = cross_val_score(estimator=dt,
                     X=x_test, y=y_test,
                     cv=gl.cv, scoring='accuracy')                       

    return getCompareParameters("Decision Trees", y_test, y_predict, scores)


def calculateLogisticRegression(x_train, x_test, y_train, y_test):
    # One-vs-the-rest (OvR) multiclass strategy.
    # Also known as one-vs-all, this strategy consists in fitting one classifier per class.
    # For each classifier, the class is fitted against all the other classes
    model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000)

    # Create L1 & L2 regularization penalty
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter distribution using uniform distribution
    # A uniform continuous random variable. This distribution is constant between loc and loc + scale.
    # C = uniform(loc=0, scale=4)
    C = np.random.uniform(0, 4, 1000)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)

    # Create randomized search 10-fold cross validation and 1000 iterations
    cv = 10
    SearchCV_Output = RandomizedSearchCV(model, hyperparameters, random_state=1, n_iter=1000, cv=gl.cv, verbose=gl.verboselevel,
                                         n_jobs=-1)
    # Fit randomized search
    best_model = SearchCV_Output.fit(x_train, y_train)
    prediction_man = best_model.predict(x_test)

    # Normal Fitter
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    scores = cross_val_score(estimator=best_model,
                     X=x_test, y=y_test,
                     cv=gl.cv, scoring='accuracy')  
    if gl.printing:
        print("Best Output Score: ", best_model.best_score_, " using Parameters: ", best_model.best_params_)
        print('The accuracy of the Logistic Regression Manually set is', metrics.accuracy_score(prediction_man, y_test))
        print('The accuracy of the Logistic Regression Automatically set is',
              metrics.accuracy_score(y_predict, y_test))
              
    # TODO add scores
              
    return getCompareParameters("Logistic Regression", y_test, y_predict, scores)


def calculateKNearestNeighbors(x_train, x_test, y_train, y_test):
    # 5.2. KNN
    # n_neighbors: Number of neighbors to use by default for k_neighbors queries
    def floatingDecimals(f_val, dec=3):
        prc = "{:." + str(dec) + "f}"  # first cast decimal as str
        #     print(prc) #str format output is {:.3f}
        return float(prc.format(f_val))

    # variables
    error = []
    test_scores = []
    train_scores = []
    k_range = 30
    neighbors = range(1, k_range)
    cv = 10  # Cross Validation

    # p=2 is equivalent to euclidean distance
    # n_jobs =-1  ->  means using all processors
    model_KNN = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski', metric_params=None,
                                     n_jobs=-1, p=2, weights='uniform')

    param_grid = dict(n_neighbors=neighbors)

    # Create randomized search 10-fold cross validation and 100 iterations
    KNN_GridSearch = GridSearchCV(model_KNN, param_grid, cv=gl.cv, n_jobs=-1, verbose=gl.verboselevel)

    # Fit randomized search
    KNN_best_model = KNN_GridSearch.fit(x_train, y_train)

    Prediction_KNN = KNN_best_model.predict(x_test)

    # Compare with Normal Fitter
    model_KNN.fit(x_train, y_train)
    y_predict = model_KNN.predict(x_test)

    # Calculating the mean error for each k value between 1 and 30 and add to the list
    for i in range(1, k_range):
        knn_model_i = KNeighborsClassifier(n_neighbors=i)
        knn_model_i.fit(x_train, y_train)
        predicted_i = knn_model_i.predict(x_test)
        error.append(np.mean(predicted_i != y_test))

    if gl.savePlots:
        # Plot mean error
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, k_range), error, color='red', marker='x', markerfacecolor='red', markersize=12)
        plt.title('Error Rate K Value')
        plt.xlabel('k')
        plt.ylabel('Mean Error')

        filePathName = gl.plotFilePath + "KNN_Mean_Error" + gl.fileFormat
        gl.savePlots(filePathName, plt)

    # Comparing Results with different k's and trained & tested data
    for i in range(1, k_range):
        knn = KNeighborsClassifier(i)
        knn.fit(x_train, y_train)
        train_scores.append(knn.score(x_train, y_train))
        test_scores.append(knn.score(x_test, y_test))

    # This score results from testing, with the same Points, which were used to train the model
    max_trained = max(train_scores)
    train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_trained]

    # score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
    max_tested = max(test_scores)
    test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_tested]

    if gl.printing:
        print("Best Output Score: ", KNN_best_model.best_score_, " using the following Parameters: ",
              KNN_best_model.best_params_)
        print('Prediction on test set is:', floatingDecimals((y_test == Prediction_KNN).mean(), 7))
        print('The accuracy of the Logistic Regression is', metrics.accuracy_score(y_predict, y_test))
        print('Maximum trained score: {} % with k equals to {}'.format(max_trained * 100,
                                                                       list(map(lambda x: x + 1, train_scores_ind))))
        print('Max test score {} % and k = {}'.format(max_tested * 100, list(map(lambda x: x + 1, test_scores_ind))))

    if gl.savePlots:
        # Result Visualisation
        plt.figure(figsize=(12, 8))
        plot_compare = sns.lineplot(x=range(1, k_range), y=train_scores, marker='*', label='Trained Score')
        plot_compare = sns.lineplot(x=range(1, k_range), y=test_scores, marker='x', label='Tested Score')

        filePathName = gl.plotFilePath + "KNN_Results" + gl.fileFormat
        gl.savePlots(filePathName, plt)
        
    scores = cross_val_score(estimator=model_KNN,
                     X=x_test, y=y_test,
                     cv=gl.cv, scoring='accuracy')      

    return getCompareParameters("K Nearest Neighbors", y_test, y_predict, scores)

def mean(list):
    return sum(list) / len(list)

def getCompareParameters(name, y_test, y_predict, scores) -> object:
    confusion = metrics.confusion_matrix(y_test, y_predict)

    # extract values from matrix
    tp = confusion[1, 1]  # true positive
    tn = confusion[0, 0]  # true negative
    fp = confusion[0, 1]  # false positive
    fn = confusion[1, 0]  # false negative

    # model evaluation
    #accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = metrics.f1_score(y_test, y_predict)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    if gl.savePlots:
        ylabel = ["Actual [Non-Diab]", "Actual [Diab]"]
        xlabel = ["Pred [Non-Diab]", "Pred [Diab]"]
        plt.figure(figsize=(15, 6))
        sns.heatmap(confusion, annot=True, xticklabels=xlabel, yticklabels=ylabel, linecolor='white', linewidths=1)

        filePathName = gl.plotFilePath + name + "Model_Evaluation" + gl.fileFormat
        gl.savePlots(filePathName, plt)

    if gl.printing:
        print(name, ' MODEL EVALUATION')
        print('Accuracy : ', mean(scores))
        print('F1 : ', f1)
        print('Specificity : ', specificity)
        print('Sensitivity : ', sensitivity, '\n\n')

    return name, mean(scores), f1, specificity, sensitivity, scores
