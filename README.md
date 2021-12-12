DiabetesPrediction Project
=======================================
Diabetes prediction with machine learning alogirthms based on scikit-learn. 

# Introduction
Group Project for diabetes prediction on the Pima Indians Diabetes Database (https://www.kaggle.com/uciml/pima-indians-diabetes-database). The following machine learning models are tested and compared:
- Decision Tree
- K-Nearest Neighbours
- Logistic Regression
- Naïve Bayes
- Support Vector Machine
- Stochastic Gradient Descent


# Installing Required packages
install the following packages:
```python
python3 -m pip install -U sklearn seaborn matplotlib numpy time sys pandas
```
# File Structure
    .
    ├── src                     # Python files 
    │   ├── algorithms.py       # Algorithms implementations
    │   ├── dataPreperation.py  # Data preperation  
    │   ├── diabetes.py         # Main file
    │   └── globalVariables.py  # Global variables and functions
    ├── docs                    # Documentation files (latex)
    ├── output                  # Output directory
    ├── data                    # Dataset directory
    ├── LICENSE
    └── README.md

# Changeable Parameters
The file globalVariables.py allows to change parameters for the entire project, following the list of parameters:
```
filePath = "../data/diabetes.csv"       # DataSet path
fileDelimiter = ","                     # DataSet delimiter
logFilePath = "../output/output.txt"    # Logfile path
plotFilePath = "../output/"             # Output folder for plots and logfile
fileFormat = ".pdf"                     # File format of the plots
savePlots = True                        # If True save the plots to files
printToFile = False                     # If True redirect stdout to logFilePath
testSize = 0.25                         # Using 0.xx of the dataset for testing
randomState = 42                        # Random Seed
cv = 5                                  # Cross validation amount of folding
verboselevel = 0                        # Verbosity output for gridsearch 0 -> none 4 -> max
```

# Run the Code
To run the code clone the git repository and execute the file diabetes.py in its directory. Depending on the global variables the plots and output will be saved in the output folder.

# Run Time
The code has a runtime of around 1 to 2 min.
