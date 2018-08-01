'''
Sunny Harjani
'''

from bs4 import BeautifulSoup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import timeit
import pickle
import warnings
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier

trainingSplit = .7
xSegments = 3
ySegments = 3
resubstitution = False # Test using training set

'''
Load .inkml files into a dictionary of lists
Dictionary key = ground truth
List = [UI, x's, y's]
'''
def loadSymbols(trainingSymbolsDir):
    if os.path.isfile("symbols.pickle"):
        symbols = pickle.load(open("symbols.pickle", "rb"))
    else:
        # Build ground truth dictionary to cross reference
        iso_GT = {}
        with open(trainingSymbolsDir+"iso_GT.txt") as f:
            for line in f:
                (key, val) = line.strip().split(',')
                iso_GT[key] = val

        symbols = defaultdict(list)
        for trainingSymbolsFile in os.listdir(trainingSymbolsDir):
            if trainingSymbolsFile.endswith(".inkml"):
                soup = BeautifulSoup(open(trainingSymbolsDir+trainingSymbolsFile), "lxml-xml")
                UI = soup.find('annotation', {'type':'UI'}).get_text()
                label = iso_GT[UI]
                traces = soup.find_all('trace')
                x = []
                y = []
                for trace in traces:
                    points = [(x).strip().split() for x in trace.get_text().split(',')]
                    x.extend([float(i[0]) for i in points])
                    y.extend([-1*float(i[1]) for i in points])
                symbols[label].append([UI, x, y])
        pickle.dump(symbols, open("symbols.pickle", "wb"))
    return symbols

'''
Represents features extracted from .inkml files
'''
class CROHMEData:
    def __init__(self):
        # Number of samples represented
        self.size = 0
        # Array containing features, as defined by extractFeatures(). Shape = [features per symbol, # of samples]
        self.features = None
        # Corresponding UIs of each sample. Shape = [1, # of samples]
        self.UIs = None
        # Corresponsing ground truth classes of each sample. Shape = [1, # of samples]
        self.classes = None

'''
Returns slopes as a result of segmented regression
x/ySegments determines how many segments
symbol[0] = symbol UI (optional)
symbol[1] = list of x coordinates
symbol[2] = list of y coordinates
'''
def extractFeatures(xSegments, ySegments, symbol):
    xMax = max(symbol[1])
    xMin = min(symbol[1])
    yMax = max(symbol[2])
    yMin = min(symbol[2])
    xInterval = (xMax - xMin) / xSegments
    yInterval = (yMax - yMin) / ySegments

    xBounds = [xMin]
    for i in range(1, xSegments+1):
        xBounds.append(xBounds[i-1] + xInterval)
    yBounds = [yMin]
    for i in range(1, ySegments+1):
        yBounds.append(yBounds[i-1] + yInterval)

    # Cluster x and y coordinates based on segmentation lines
    quadrants = [[[] for y in range(ySegments)] for x in range(xSegments)]
    for xPoint, yPoint in zip(symbol[1], symbol[2]):
        foundQuadrant = False
        for xBoundIndex in range(1, xSegments+1):
            if (foundQuadrant):
                break
            for yBoundIndex in range(1, ySegments+1):
                if (xPoint < xBounds[xBoundIndex] and yPoint < yBounds[yBoundIndex]):
                    quadrants[xBoundIndex-1][yBoundIndex-1].append([xPoint, yPoint])
                    foundQuadrant = True
                    break

    # Calculate linear regression in every quadrant
    warnings.simplefilter('ignore', np.RankWarning)
    for r in range(len(quadrants)):
        for c in range(len(quadrants[r])):
            if len(quadrants[r][c]) > 1:
                x = [point[0] for point in quadrants[r][c]]
                y = [point[1] for point in quadrants[r][c]]
                m, b = np.polyfit(x, y, 1)
                quadrants[r][c] = m # Discard y-intercept

                # Visualize segmented regression
                # p = np.poly1d([m, b])
                # plt.plot(x, p(x))
            else: # Assume slope = 0 if < 2 points exist in the quadrant
                quadrants[r][c] = 0

    # Visualize gridlines
    # for xBound in xBounds:
    #     plt.axvline(x=xBound, color='r')
    # for yBound in yBounds:
    #     plt.axhline(y=yBound, color='r')

    # Visualize raw input
    # plt.scatter(symbol[1], symbol[2])

    # Visualize smoothed input
    # plt.plot(symbol[1], symbol[2])

    # Uncomment for visualization
    # plt.axis('equal')
    # print(symbol[0])
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    return np.array(quadrants).flatten()

'''
Segments symbols by feature and stores them as CROHMEData objects
'''
def loadData():
    if os.path.isfile("trainingData.pickle"): # Check if I've done this before
        trainingData = pickle.load(open("trainingData.pickle", "rb"))
        testingData = pickle.load(open("testingData.pickle", "rb"))
    else:
        symbols = loadSymbols(sys.argv[1])
        # symbols = loadSymbols("E:/Sunny/Downloads/task2-trainSymb2014/training&junk/")
        trainingData = CROHMEData()
        testingData = CROHMEData()
        for key in symbols.keys():
            numOfTrainingSymbols = int(len(symbols[key]) * trainingSplit)
            trainingData.size = trainingData.size + numOfTrainingSymbols
            testingData.size = testingData.size + len(symbols[key]) - numOfTrainingSymbols

        featuresSize = xSegments * ySegments
        trainingData.features = np.zeros((trainingData.size, featuresSize))
        trainingData.UIs = np.zeros(trainingData.size, dtype=object)
        trainingData.classes = np.zeros(trainingData.size, dtype=object)
        trainingIndex = 0

        testingData.features = np.zeros((testingData.size, featuresSize))
        testingData.UIs = np.zeros(testingData.size, dtype=object)
        testingData.classes = np.zeros(testingData.size, dtype=object)
        testingIndex = 0

        # Split training/test samples at every class to retain prior probabilities
        for key in symbols.keys():
            symbolSplitIndex = int(len(symbols[key]) * trainingSplit)
            symbolIndex = 0
            for symbol in symbols[key]:
                features = extractFeatures(xSegments, ySegments, symbol)
                if (symbolIndex < symbolSplitIndex):
                    trainingData.features[trainingIndex] = features
                    trainingData.UIs[trainingIndex] = symbol[0]
                    trainingData.classes[trainingIndex] = key
                    trainingIndex = trainingIndex + 1
                else:
                    testingData.features[testingIndex] = features
                    testingData.UIs[testingIndex] = symbol[0]
                    testingData.classes[testingIndex] = key
                    testingIndex = testingIndex + 1
                symbolIndex = symbolIndex + 1
        # Hope I don't have to do it again
        pickle.dump(trainingData, open("trainingData.pickle", "wb"))
        pickle.dump(testingData, open("testingData.pickle", "wb"))
    return trainingData, testingData

'''
Returns a given array with duplicates removed
'''
def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

if __name__ == "__main__":
    trainingData, testingData = loadData()
    if (resubstitution):
        testingData = trainingData

    if os.path.isfile("KDTree.pickle"):
        tree = pickle.load(open("KDTree.pickle", "rb"))
    else:
        tree = KDTree(trainingData.features)
        pickle.dump(tree, open("KDTree.pickle", "wb"))

    matches = 0
    classificationFile = open("KDclassification.txt", "w")
    groundTruthFile = open("groundTruth.txt", "w")
    for testingFeature, testingClass, testingUI in zip(testingData.features, testingData.classes, testingData.UIs):
        # Query KDTree for increasing number of neighbors until 10 unique neighbors are found
        numOfGuesses = 1
        guesses = []
        while (len(guesses) < 10):
            numOfGuesses = numOfGuesses * 10
            dist, ind = tree.query([testingFeature], k=numOfGuesses)
            guesses = unique(trainingData.classes[ind])
        guesses = guesses[:10]
        classificationFile.write("{}, {}\n".format(testingUI, ", ".join(guesses)))
        groundTruthFile.write("{}, {}\n".format(testingUI, testingClass))
        # Internal accuracy measure
        if (guesses[0] == testingClass):
            matches = matches + 1
    classificationFile.close()
    groundTruthFile.close()
    # print ('KD', matches/testingData.size)

    if os.path.isfile("RandomForest.pickle"):
        clf = pickle.load(open("RandomForest.pickle", "rb"))
    else:
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(trainingData.features, trainingData.classes)
        pickle.dump(clf, open("RandomForest.pickle", "wb"))
    matches = 0
    classificationFile = open("RFclassification.txt", "w")
    for testingFeature, testingClass, testingUI in zip(testingData.features, testingData.classes, testingData.UIs):
        # Predict probabilities for every class
        guesses = clf.predict_proba(testingFeature.reshape(1, -1))[0]
        # Keep the 10 biggest probabilities
        indices = np.argsort(-guesses)[:10]
        guesses = clf.classes_[indices]
        # Internal accuracy measure
        if (guesses[0] == testingClass):
            matches = matches + 1
        classificationFile.write("{}, {}\n".format(testingUI, ", ".join(guesses)))
    classificationFile.close()
    # print ('RF', matches/testingData.size)