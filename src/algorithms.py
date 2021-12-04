#!/usr/bin/env python3
import time
import globalVariables as gl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns  # statistical plot
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier  # K Nearest Neighbors
from sklearn.svm import SVC  # Support Vector Machines
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn import tree  # Decision Tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
import warnings

warnings.filterwarnings('ignore')


def calculateStochasticGradientDescent(x_train, x_test, y_train, y_test):
    sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    sgd.fit(x_train, y_train)
    y_predict = sgd.predict(x_test)

    return getCompareParameters("Stochastic Gradient Descent", y_test, y_predict)


def calculateSupportVectorMachines(x_train, x_test, y_train, y_test):
    # Todo use different kernels
    support_vector_classifier = SVC(kernel="linear").fit(x_train, y_train)

    # print(support_vector_classifier)

    y_predict = support_vector_classifier.predict(x_test)

    # Model Tuning & Validation
    scores = cross_val_score(estimator=support_vector_classifier,
                             X=x_train, y=y_train,
                             cv=10, scoring='f1_macro')  # means we have a k-fold of 10
    if gl.printing:
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    # support_vector_classifier.predict(x_test)[:10]
    # svm_params = {"C": np.arange(1, 20)}
    #
    # svm = SVC(kernel="linear")
    # svm_cv = GridSearchCV(svm, svm_params, cv=8)
    #
    # start_time = time.time()
    #
    # svm_cv.fit(x_train, y_train)
    #
    # elapsed_time = time.time() - start_time
    #
    # svm_tuned = SVC(kernel="linear", C=2).fit(x_train, y_train)
    #
    # if gl.printing:
    #     print(f"Elapsed time for Support Vector Regression cross validation: " f"{elapsed_time:.3f} seconds")
    #     # best score
    #     print(svm_cv.best_score_)
    #     # best parameters
    #     print(svm_cv.best_params_)
    #     print(svm_tuned)
    #
    # y_predict = svm_tuned.predict(x_test)
    # cm = confusion_matrix(y_test, y_predict)
    #
    # if gl.printing:
    #     print(cm)
    #     print("Our Accuracy is: ", (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))
    #     print(accuracy_score(y_test, y_predict))
    #     print(classification_report(y_test, y_predict))

    return getCompareParameters("Support Vector Machines", y_test, y_predict)


def calculateNaiveBayes(x_train, x_test, y_train, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)  # estimated targets as returned by the classifier

    return getCompareParameters("Naive Bayes", y_test, y_predict)


def calculateDecisionTrees(x_train, x_test, y_train, y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)  # estimated targets as returned by the classifier

    return getCompareParameters("Decision Trees", y_test, y_predict)


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
    SearchCV_Output = RandomizedSearchCV(model, hyperparameters, random_state=1, n_iter=1000, cv=cv, verbose=0,
                                         n_jobs=-1)
    # Fit randomized search
    best_model = SearchCV_Output.fit(x_train, y_train)
    prediction_man = best_model.predict(x_test)

    # Normal Fitter
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    if gl.printing:
        print("Best Output Score: ", best_model.best_score_, " using Parameters: ", best_model.best_params_)
        print('The accuracy of the Logistic Regression Manually set is', metrics.accuracy_score(prediction_man, y_test))
        print('The accuracy of the Logistic Regression Automatically set is',
              metrics.accuracy_score(y_predict, y_test))

    return getCompareParameters("Logistic Regression", y_test, y_predict)


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
    KNN_GridSearch = GridSearchCV(model_KNN, param_grid, cv=cv, n_jobs=-1)

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

    return getCompareParameters("K Nearest Neighbors", y_test, y_predict)


def getCompareParameters(name, y_test, y_predict) -> object:
    confusion = metrics.confusion_matrix(y_test, y_predict)

    # extract values from matrix
    tp = confusion[1, 1]  # true positive
    tn = confusion[0, 0]  # true negative
    fp = confusion[0, 1]  # false positive
    fn = confusion[1, 0]  # false negative

    # model evaluation
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
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
        print('Accuracy : ', accuracy)
        print('Precision : ', precision)
        print('Recall : ', recall)
        print('F1 : ', f1)
        print('Specificity : ', specificity)
        print('Sensitivity : ', sensitivity, '\n\n')

    return name, accuracy, f1, specificity, sensitivity
