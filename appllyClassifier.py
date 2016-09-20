__author__ = 'm.baytel'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import datetime
import pickle
import os
import idx2numpy
import common


def diffList(list1,list2):
    i=0
    res=[]
    for a,b in zip(list1,list2):
        if a!=b:
            res.append(i)
        i+=1
    return res


def saveErrors(indexList,pictureList,realLabel,predictLabel,dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    for index in indexList:
        picArray = pictureList[index]

        picture = picArray.reshape(28,28)

        im = Image.fromarray(~picture,'L')
        im.save(dirPath + os.sep +str(realLabel[index])+'_'+str(predictLabel[index])+ '_' + str(index)+'.png','png')



#methodeName - knn,svm,tree
#labelOfData - a text for folder name (in order to distinguish data, pca data, deskew  data etc)
#listMethodeParams - a list of disct (parameters for method are packed in dictionary)
def appllyClassifier(methodName,labelOfData,trainDataPath,trainLabelsPath,testDataPath,testLabelsPath,listMethodParams,limitTrainSize,needInverseData,out_dir,needSaveError):
    trainData    =  idx2numpy.convert_from_file(trainDataPath)  #0 means black usually! And 255 means white!
    trainLabels  =  idx2numpy.convert_from_file(trainLabelsPath)

    testData    =  idx2numpy.convert_from_file(testDataPath)    #0 means black usually! And 255 means white!
    testLabels  =  idx2numpy.convert_from_file(testLabelsPath)

    if needInverseData:
        trainData = ~trainData
        testData  = ~testData

    if len(trainData) > limitTrainSize :
        trainData   = trainData[0:limitTrainSize]
        trainLabels = trainLabels[0:limitTrainSize]

    shape = trainData.shape
    if (len(shape) == 3):
        trainData=trainData.reshape(shape[0],shape[1]*shape[2])

    shape = testData.shape
    if (len(shape) == 3):
        testData=testData.reshape(shape[0],shape[1]*shape[2])


    classyfier=None

    dirName = out_dir + os.sep + labelOfData + os.sep + methodName

    if not os.path.exists(dirName):
        os.makedirs(dirName)

    trainError = []
    testError =  []

    i=0
    for param in listMethodParams:

        print(methodName,param,datetime.datetime.now())

        if methodName=='knn':
            classyfier = KNeighborsClassifier(**param)
        if methodName=='svm':
            classyfier = LinearSVC(**param)
        if methodName=='tree':
            classyfier = DecisionTreeClassifier(**param)


        classyfier.fit(trainData,trainLabels)

        print(methodName,datetime.datetime.now(),'1')

        trainRes = classyfier.predict(trainData)

        print(methodName,datetime.datetime.now(),'2')

        testRes = classyfier.predict(testData)

        print(methodName,datetime.datetime.now(),'3')

        diffTrain = diffList(trainRes,trainLabels)
        diffTest  = diffList(testRes,testLabels)

        if needSaveError:
            saveErrors(diffTrain,trainData,trainLabels,trainRes,dirName+os.sep+'train_errors_'+str(i))
            saveErrors(diffTest,testData,testLabels,testRes,dirName+os.sep+'test_errors_'+str(i))

        trainError.append(len(diffTrain)/len(trainLabels))
        testError.append(len(diffTest)/len(testLabels))

        print(methodName,datetime.datetime.now(),'4')

        i = i+1

    pickle.dump(listMethodParams,open(dirName + os.sep + 'parameters','wb'))
    pickle.dump(trainError,open(dirName + os.sep + 'trainerror','wb'))
    pickle.dump(testError,open(dirName + os.sep + 'testerror','wb'))

    print(methodName,labelOfData,' train error',trainError)
    print(methodName,labelOfData,' test error',testError)


common_params={'out_dir':'./out','limitTrainSize':60000,'trainLabelsPath':common.train_labels,'testLabelsPath':common.test_labels}


svn_params=[{'intercept_scaling':1/1000},{'intercept_scaling':1/100},{'intercept_scaling':1/10},{'intercept_scaling':1},{'intercept_scaling':10},{'intercept_scaling':100},{'intercept_scaling':1000}]
svn_args={'methodName':'svm','listMethodParams':svn_params}


tree_args={'methodName':'tree','listMethodParams':[{}]}


knn_params = [ {'n_neighbors':1}, {'n_neighbors':3}, {'n_neighbors':5}, {'n_neighbors':7}, {'n_neighbors':9}, {'n_neighbors':11}]
knn_args={'methodName':'knn','listMethodParams':knn_params}


usualData={'labelOfData':'usual','trainDataPath':common.train_images,'testDataPath':common.test_images,'needInverseData':False,'needSaveError':True}
deskewData={'labelOfData':'deskew','trainDataPath':common.train_deskew_images,'testDataPath':common.test_deskew_images,'needInverseData':False,'needSaveError':True}

usualPCAData={'labelOfData':'pca-usual','trainDataPath':common.train_images_pca,'testDataPath':common.test_images_pca,'needInverseData':False,'needSaveError':False}
deskewPCAData={'labelOfData':'pca-deskew','trainDataPath':common.train_deskew_pca_images,'testDataPath':common.test_deskew_pca_images,'needInverseData':False,'needSaveError':False}


dirList = [usualData,deskewData,usualPCAData,deskewPCAData]
methodList = [svn_args,tree_args,knn_args]

for methodArgs in methodList:
    for dirs in dirList:
        params={}
        params.update(common_params)
        params.update(methodArgs)
        params.update(dirs)

        appllyClassifier(**params)
