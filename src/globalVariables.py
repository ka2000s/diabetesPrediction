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

# User variables to modify
filePath = "../data/diabetes.csv"  # DataSet
fileDelimiter = ","  # DataSet Delimiter
logFilePath = "../output/output.txt"  # Logfile Path
plotFilePath = "../output/"  # File Path for plots
fileFormat = ".pdf"  # File format of the plots
savePlots = True  # If True save the plots to files
printing = True  # If True print files (either to console or to files)
printToFile = True  # If true redirect stdout to logFilePath  (printing needs to be true as well to work)
testSize = 0.30  # STILL TO DECIDE
randomState = 1  # STILL TO DECIDE


# Results class for storing the results to compare
class Results(object):
    def __init__(self, name_, accuracy_, f1_, specificity_, sensitivity_):
        self.name = name_
        self.accuracy = accuracy_
        self.f1 = f1_
        self.specificity = specificity_
        self.sensitivity = sensitivity_


def savePlots(path, plot):
    plot.savefig(path, bbox_inches="tight", pad_inches=0.3, transparent=False, dpi=600)
