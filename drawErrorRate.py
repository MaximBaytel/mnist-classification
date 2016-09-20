__author__ = 'm.baytel'

import matplotlib.pyplot as plt
import pickle
import common
import matplotlib.patches as mpatches
import os


#def plotEigens(dumpFile,title):


def plotErrorRate(trainfile,testfile,title):
    trainratio = pickle.load(open(trainfile,'rb'))
    testratio = pickle.load(open(testfile,'rb'))

    plt.ylabel('error rate')
    ax = plt.gca()
    ax.grid(True)
    green_patch  = mpatches.Patch(color='green', label='test data')
    blue_patch = mpatches.Patch(color='blue', label='train data')
    plt.legend(handles=[green_patch,blue_patch])
    plt.plot(trainratio)
    plt.plot(testratio)
    plt.title(title)
    plt.show()

labelList = ['usual','deskew','pca-usual','pca-deskew']
titleList = ['original data','deskew data', 'data after pca reduction','deskew data after pca reduction']

methodNames = ['svm','knn']

for label,title in zip(labelList,titleList):
    for name in methodNames:
        trainpath = common.out_dir + os.sep + label + os.sep + name + os.sep + 'trainerror'
        testpath  = common.out_dir + os.sep + label + os.sep + name + os.sep + 'testerror'

        plotErrorRate(trainpath,testpath,name + ':' + title)