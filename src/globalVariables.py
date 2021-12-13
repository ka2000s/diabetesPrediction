#!/usr/bin/env python3
global filePath
global fileDelimiter
global logFilePath
global plotFilePath
global fileFormat
global showPlots
global savePlots
global printing
global printToFile
global testSize
global randomState

# Changeable parameters for entire project
filePath = "../data/diabetes.csv"       # DataSet path
fileDelimiter = ","                     # DataSet delimiter
logFilePath = "../output/output.txt"    # Logfile path
plotFilePath = "../output/"             # Output folder for plots and logfile
fileFormat = ".pdf"                     # File format of the plots
savePlots = True                        # If True save the plots to files
printToFile = True                      # If True redirect stdout to logFilePath
testSize = 0.25                         # Using 0.xx of the dataset for testing
randomState = 42                        # Random Seed
cv = 5                                  # Cross validation amount of folding
verboselevel = 0                        # Verbosity output for grid search 0 -> none 4 -> max

# Results class for storing the results to compare
class Results(object):
    def __init__(self, name_, accuracy_, f1_, specificity_, sensitivity_):
        self.name = name_
        self.accuracy = accuracy_
        self.f1 = f1_
        self.specificity = specificity_
        self.sensitivity = sensitivity_
        

def savePlots(path, plot):
    plot.savefig(path, bbox_inches="tight", pad_inches=0.3, transparent=False, dpi=300)
