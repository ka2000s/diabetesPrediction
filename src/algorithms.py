#!/usr/bin/env python3
import time
import globalVariables as gl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import seaborn as sns  # statistical plot
from sklearn.linear_model import SGDClassifier
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier  # K Nearest Neighbors
from sklearn.svm import SVC  # Support Vector Machines
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV

warnings.filterwarnings("ignore")


def calculateStochasticGradientDescent(x_train, x_test, y_train, y_test):
    # Test Parameters to evaluate
    testParameters = [{
        "loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber',
                 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        "penalty": ['l2', 'l1', 'elasticnet'], "class_weight": ['balanced'], "l1_ratio": np.linspace(0, 1, num=10),
        "alpha": np.power(10, np.arange(-4, 1, dtype=float))
    }]
    best_parameters = []
    best_score = []
    # Scaling data
    Std_scaler = StandardScaler()
    x_train_scaled = Std_scaler.fit_transform(x_train)
    x_test_scaled = Std_scaler.fit_transform(x_test)
    print("Stochastic Gradient Descent GridSeacrh on scaled an unscaled dataset")
    for trainingSet in [x_train_scaled, x_train]:
        sgd = SGDClassifier(random_state=True)
        sgd_cv = GridSearchCV(sgd, testParameters, cv=gl.cv, verbose=gl.verboselevel, n_jobs=-1,
                              scoring='balanced_accuracy')
        sgd_cv.fit(trainingSet, y_train)

        best_parameters.append(sgd_cv.best_params_)
        best_score.append(sgd_cv.best_score_)

    if best_score[0] >= best_score[1]:
        # Scaled Dataset
        sgd = SGDClassifier(**best_parameters[0])
        sgd.fit(x_train_scaled, y_train)
        y_predict = sgd.predict(x_test_scaled)
        print("Scaled Model has better accuracy of: ", best_score[0], " vs: ", best_score[1])
        print("Best Parameters are: ", best_parameters[0])
        accuracyScores = cross_val_score(estimator=sgd,
                                         X=x_test_scaled, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=sgd,
                                   X=x_test, y=y_test,
                                   cv=gl.cv, scoring='f1')
    else:
        # Non Scaled Dataset
        sgd = SGDClassifier(**best_parameters[1])
        sgd.fit(x_train, y_train)
        y_predict = sgd.predict(x_test)
        print("Non scaled Model has better accuracy of: ", best_score[1], " vs: ", best_score[0])
        print("Best Parameters are: ", best_parameters[1])
        accuracyScores = cross_val_score(estimator=sgd,
                                         X=x_test, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=sgd,
                                   X=x_test, y=y_test,
                                   cv=gl.cv, scoring='f1')

    return getCompareParameters("Stochastic Gradient Descent", y_test, y_predict, accuracyScores, f1Scores)


def calculateSupportVectorMachines(x_train, x_test, y_train, y_test):
    # Test Parameters to evaluate
    testParameters = [
        {"kernel": ["linear"], "C": [0.9, 1, 2, 10], "class_weight": ['balanced']},
        {"kernel": ["rbf"], "gamma": ['scale', 'auto'], "C": [0.9, 1, 2, 10], "class_weight": ['balanced']},
        {"kernel": ["poly"], "gamma": ['scale'], "C": [0.9, 1, 2, 10], "degree": [2, 3, 4, 5, 6, 7, 8], "coef0": [0.0],
         "class_weight": ['balanced']},
        {"kernel": ["sigmoid"], "gamma": ['scale', 'auto'], "C": [0.9, 1, 2, 10], "coef0": [0.0],
         "class_weight": ['balanced']},
    ]
    best_parameters = []
    best_score = []
    # Scaling data
    Std_scaler = StandardScaler()
    x_train_scaled = Std_scaler.fit_transform(x_train)
    x_test_scaled = Std_scaler.fit_transform(x_test)
    print("Support Vector Machines GridSearch on scaled an unscaled dataset")
    for trainingSet in [x_train_scaled, x_train]:
        # Set the parameters by cross-validation
        svm = SVC(cache_size=1000)
        svm_cv = GridSearchCV(svm, testParameters, scoring='balanced_accuracy', cv=gl.cv, verbose=gl.verboselevel,
                              n_jobs=-1)
        svm_cv.fit(trainingSet, y_train)

        best_parameters.append(svm_cv.best_params_)
        best_score.append(svm_cv.best_score_)

    if best_score[0] >= best_score[1]:
        # Scaled Dataset
        svm = SVC(**best_parameters[0])
        svm.fit(x_train_scaled, y_train)
        y_predict = svm.predict(x_test_scaled)
        print("Scaled Model has better accuracy of: ", best_score[0], " vs: ", best_score[1])
        print("Best Parameters are: ", best_parameters[0])
        accuracyScores = cross_val_score(estimator=svm,
                                         X=x_test_scaled, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=svm,
                                   X=x_test_scaled, y=y_test,
                                   cv=gl.cv, scoring='f1')
    else:
        # Non Scaled Dataset
        svm = SVC(**best_parameters[1])
        svm.fit(x_train, y_train)
        y_predict = svm.predict(x_test)
        print("Non scaled Model has better accuracy of: ", best_score[1], " vs: ", best_score[0])
        print("Best Parameters are: ", best_parameters[1])
        accuracyScores = cross_val_score(estimator=svm,
                                         X=x_test, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=svm,
                                   X=x_test, y=y_test,
                                   cv=gl.cv, scoring='f1')

    return getCompareParameters("Support Vector Machines", y_test, y_predict, accuracyScores, f1Scores)


def calculateNaiveBayes(x_train, x_test, y_train, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    accuracyScores = cross_val_score(estimator=model,
                                     X=x_test, y=y_test,
                                     cv=gl.cv, scoring='accuracy')
    f1Scores = cross_val_score(estimator=model,
                               X=x_test, y=y_test,
                               cv=gl.cv, scoring='f1')

    return getCompareParameters("Naive Bayes", y_test, y_predict, accuracyScores, f1Scores)


def calculateDecisionTrees(x_train, x_test, y_train, y_test):
    # Test Parameters to evaluate
    testParameters = [
        {"criterion": ['gini', 'entropy'], "splitter": ['best', 'random'],
         "max_features": ['auto', 'sqrt', 'log2'],
         "class_weight": ['balanced']}
    ]
    dt = DecisionTreeClassifier(random_state=gl.randomState)
    dt_cv = GridSearchCV(dt, testParameters, scoring='balanced_accuracy', cv=gl.cv, verbose=gl.verboselevel, n_jobs=-1)
    dt_cv.fit(x_train, y_train)
    # Use the best parameters
    print("Best Parameters are: ", dt_cv.best_params_)
    dt = DecisionTreeClassifier(**dt_cv.best_params_)
    dt.fit(x_train, y_train)
    y_predict = dt.predict(x_test)  # estimated targets as returned by the classifier
    accuracyScores = cross_val_score(estimator=dt,
                                     X=x_test, y=y_test,
                                     cv=gl.cv, scoring='accuracy')
    f1Scores = cross_val_score(estimator=dt,
                               X=x_test, y=y_test,
                               cv=gl.cv, scoring='f1')

    return getCompareParameters("Decision Trees", y_test, y_predict, accuracyScores, f1Scores)


def calculateLogisticRegression(x_train, x_test, y_train, y_test):
    # variables
    uni_points = 30
    best_parameters = []
    best_score = []
    # Scaling data
    Std_scaler = StandardScaler()
    x_train_scaled = Std_scaler.fit_transform(x_train)
    x_test_scaled = Std_scaler.fit_transform(x_test)

    # Test Parameters to evaluate
    testParameters = [
        {"solver": ['liblinear'], "penalty": ['l1', 'l2'], "C": np.random.uniform(0, 4, uni_points),
         "max_iter": [1000], "multi_class": ['ovr', 'auto', 'multinomial']},
        {"solver": ['lbfgs'], "penalty": ['none', 'l2'], "C": np.random.uniform(0, 4, uni_points),
         "max_iter": [1000], "multi_class": ['ovr', 'auto', 'multinomial']},
        {"solver": ['newton-cg'], "penalty": ['none', 'l2'], "C": np.random.uniform(0, 4, uni_points),
         "max_iter": [1000], "multi_class": ['ovr', 'auto', 'multinomial']},
        {"solver": ['sag'], "penalty": ['none', 'l2'], "C": np.random.uniform(0, 4, uni_points),
         "max_iter": [1000], "multi_class": ['ovr', 'auto', 'multinomial']},
        {"solver": ['saga'], "penalty": ['l1', 'l2', 'none'], "C": np.random.uniform(0, 4, uni_points),
         "max_iter": [1000], "multi_class": ['ovr', 'auto', 'multinomial']}
    ]

    for trainingSet in [x_train_scaled, x_train]:
        model_LR = LogisticRegression()
        # 5-fold cross validation randomized search with 1000 iterations
        SearchCV_Output = RandomizedSearchCV(model_LR, testParameters, random_state=gl.randomState, n_iter=1000,
                                             cv=gl.cv,
                                             verbose=gl.verboselevel, n_jobs=-1)
        # Fit randomized search
        SearchCV_Output.fit(trainingSet, y_train)

        best_parameters.append(SearchCV_Output.best_params_)
        best_score.append(SearchCV_Output.best_score_)

    if best_score[0] >= best_score[1]:
        # Scaled Dataset
        model_LR = LogisticRegression(**best_parameters[0])
        model_LR.fit(x_train_scaled, y_train)
        y_predict = model_LR.predict(x_test_scaled)
        print("Scaled Model has better accuracy of: ", best_score[0], " vs: ", best_score[1])
        print("Best Parameters are: ", best_parameters[0])
        accuracyScores = cross_val_score(estimator=model_LR,
                                         X=x_test_scaled, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=model_LR,
                                   X=x_test_scaled, y=y_test,
                                   cv=gl.cv, scoring='f1')
    else:
        # Non Scaled Dataset
        model_LR = LogisticRegression(**best_parameters[1])
        model_LR.fit(x_train, y_train)
        y_predict = model_LR.predict(x_test)
        print("Non scaled Model has better accuracy of: ", best_score[1], " vs: ", best_score[0])
        print("Best Parameters are: ", best_parameters[1])
        accuracyScores = cross_val_score(estimator=model_LR,
                                         X=x_test, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=model_LR,
                                   X=x_test, y=y_test,
                                   cv=gl.cv, scoring='f1')

    return getCompareParameters("Logistic Regression", y_test, y_predict, accuracyScores, f1Scores)


def calculateKNearestNeighbors(x_train, x_test, y_train, y_test):
    # variables
    k_range = 30
    best_parameters = []
    best_score = []

    # Scaling data
    Std_scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(Std_scaler.fit_transform(x_train))
    # overwrite scaled data by also normalizing it
    x_train_scaled = preprocessing.normalize(x_train_scaled)
    x_test_scaled = pd.DataFrame(Std_scaler.fit_transform(x_test))

    # Test different hyper-parameters to evaluate best constelation
    eval_parameters = [
        {"algorithm": ['auto', 'kd_tree', 'ball_tree', 'brute'], "n_neighbors": np.arange(1, k_range),
         "leaf_size": [10, 20, 30, 40, 50],
         "weights": ['uniform', 'distance']}
    ]
    for trainingSet in [x_train_scaled, x_train]:
        model_KNN = KNeighborsClassifier()
        # Create randomized search 5-fold cross validation
        KNN_GridSearch = GridSearchCV(model_KNN, eval_parameters, scoring='accuracy', cv=gl.cv, n_jobs=-1,
                                      verbose=gl.verboselevel)
        # Fit randomized search
        KNN_GridSearch.fit(trainingSet, y_train)

        best_parameters.append(KNN_GridSearch.best_params_)
        best_score.append(KNN_GridSearch.best_score_)

    if best_score[0] >= best_score[1]:
        # Scaled Dataset
        model_KNN = KNeighborsClassifier(**best_parameters[0])
        model_KNN.fit(x_train_scaled, y_train)
        y_predict = model_KNN.predict(x_test_scaled)
        print("Scaled Model has better accuracy of: ", best_score[0], " vs: ", best_score[1])
        print("Best Parameters are: ", best_parameters[0])
        accuracyScores = cross_val_score(estimator=model_KNN,
                                         X=x_test_scaled, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=model_KNN,
                                   X=x_test_scaled, y=y_test,
                                   cv=gl.cv, scoring='f1')
    else:
        # Non Scaled Dataset
        model_KNN = KNeighborsClassifier(**best_parameters[1])
        model_KNN.fit(x_train, y_train)
        y_predict = model_KNN.predict(x_test)
        print("Non scaled Model has better accuracy of: ", best_score[1], " vs: ", best_score[0])
        print("Best Parameters are: ", best_parameters[1])
        accuracyScores = cross_val_score(estimator=model_KNN,
                                         X=x_test, y=y_test,
                                         cv=gl.cv, scoring='accuracy')
        f1Scores = cross_val_score(estimator=model_KNN,
                                   X=x_test, y=y_test,
                                   cv=gl.cv, scoring='f1')

    return getCompareParameters("K Nearest Neighbors", y_test, y_predict, accuracyScores, f1Scores)


def mean(lst):
    return sum(lst) / len(lst)


def getCompareParameters(name, y_test, y_predict, accuracyScores, f1Scores) -> object:
    confusion = metrics.confusion_matrix(y_test, y_predict)

    # extract values from matrix
    tp = confusion[1, 1]  # true positive
    tn = confusion[0, 0]  # true negative
    fp = confusion[0, 1]  # false positive
    fn = confusion[1, 0]  # false negative

    # model evaluation
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = mean(f1Scores)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    if gl.savePlots:
        ylabel = ["Actual [Non-Diab]", "Actual [Diab]"]
        xlabel = ["Pred [Non-Diab]", "Pred [Diab]"]
        plt.figure(figsize=(15, 6))
        sns.heatmap(confusion, annot=True, xticklabels=xlabel, yticklabels=ylabel, linecolor='white', linewidths=1)

        filePathName = gl.plotFilePath + name + "Model_Evaluation" + gl.fileFormat
        gl.savePlots(filePathName, plt)

    print(name, ' MODEL EVALUATION')
    print('Accuracy : ', mean(accuracyScores))
    print('F1 : ', f1)
    print('Specificity : ', specificity)
    print('Sensitivity : ', sensitivity, '\n\n')

    return name, mean(accuracyScores), f1, specificity, sensitivity, accuracyScores, f1Scores
