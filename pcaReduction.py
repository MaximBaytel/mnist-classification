__author__ = 'm.baytel'

import idx2numpy
import numpy
import pickle
from sklearn.decomposition import PCA
from PIL import Image
import os
import common


def pcaReduction(trainIdxPath,testIdxPath,outTrainPath,outTestIdx,outRatioFile,pcaEigenDir):
    trainData    =  idx2numpy.convert_from_file(trainIdxPath)
    testData     =  idx2numpy.convert_from_file(testIdxPath)

    shape = trainData.shape
    if (len(shape) > 1):
        trainData=trainData.reshape(shape[0],shape[1]*shape[2])

    shape = testData.shape
    if (len(shape) > 1):
        testData=testData.reshape(shape[0],shape[1]*shape[2])

    pca=PCA()

    pca.fit(trainData)

    cumSumRatio = numpy.cumsum(pca.explained_variance_ratio_)

    indOf09  = numpy.argmax(cumSumRatio>=0.9)
    indOf095 = numpy.argmax(cumSumRatio>=0.95)
    indOf099 = numpy.argmax(cumSumRatio>=0.99)


    print('explained variance ratio: ',pca.explained_variance_ratio_[0:indOf099])

    print('cumsum of explained variance ratio: ',cumSumRatio[0:indOf099])

    print('indexes of 0.9, 095, 0.99 ',indOf09,indOf095,indOf099)

    #print('Shape of components:',pca.components_.shape)

    A=pca.components_[:,0:indOf095+1]

    for i in range(0,indOf095+1):
        v = A[:,i]
        v = abs(v)*255
        v=v.reshape(28,28)
        im = Image.fromarray(~v.astype('uint8'),'L')
        im.save(pcaEigenDir+os.sep+str(i)+'.png','png')

    #print('xxx',A.shape)

    trainData = numpy.dot(trainData,A)

    print('train shape',trainData.shape,trainData.dtype)

    testData  = numpy.dot(testData,A)

    print('test shape',testData.shape,trainData.dtype)

    f_write = open(outTrainPath, 'wb')
    idx2numpy.convert_to_file(f_write,trainData)

    f_write = open(outTestIdx, 'wb')
    idx2numpy.convert_to_file(f_write,testData)

    pickle.dump(cumSumRatio[0:indOf099],open(outRatioFile,'wb'))




pcaReduction(common.train_images,common.test_images,common.train_images_pca,common.test_images_pca,common.out_ratio_usual,common.eigenPCA_dir)

pcaReduction(common.train_deskew_images,common.test_deskew_images,common.train_deskew_pca_images,common.test_deskew_pca_images,common.out_ratio_deskew,common.eigenPCA_deskewdir)