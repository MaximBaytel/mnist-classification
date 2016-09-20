__author__ = 'm.baytel'

import idx2numpy
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy
import os
import sys
import datetime
import pickle


def calcMeanAndStd():
    mean_train_picture = numpy.mean(train_pictures,0)
    std_train_picture  = numpy.std(train_pictures[0:3000],0)

    train_pictures_by_classes = [[] for i in range(10)]
    train_pictures_by_classes_array = [None for i in range(10)]

    for i in range(10):
        name='./internal/train_'+str(i)
        if not os.path.exists(name):
            os.makedirs(name)

    for picture,label in zip(train_pictures,train_labels):
        #print(picture.dtype)
        number = len(train_pictures_by_classes[label])
        train_pictures_by_classes[label].append(picture)
        im = Image.fromarray(picture,'L')
        im.save('./internal/train_'+str(label)+os.sep+str(number)+'.png','png')

    for i in range(10):
        train_pictures_by_classes_array[i] = numpy.array(train_pictures_by_classes[i])

        mean_image = Image.fromarray(numpy.mean(train_pictures_by_classes_array[i],axis=0).astype('uint8'),'L')
        std_image = Image.fromarray(numpy.std(train_pictures_by_classes_array[i],axis=0).astype('uint8'),'L')
        std_image_inv = Image.fromarray(~numpy.std(train_pictures_by_classes_array[i],axis=0).astype('uint8'),'L')

        mean_image.save(out_dir+os.sep + 'mean_train_' + str(i)+'.png','png')
        std_image.save(out_dir+os.sep + 'std_train_' + str(i)+'.png','png')
        std_image_inv.save(out_dir+os.sep + 'std_train_inv_' + str(i)+'.png','png')


    im = Image.fromarray(mean_train_picture.astype('uint8'),'L')
    im.save(out_dir+os.sep +'mean_train.png','png')

    im = Image.fromarray(std_train_picture.astype('uint8'),'L')
    im.save(out_dir+os.sep +'std_train.png','png')

    im = Image.fromarray(~std_train_picture.astype('uint8'),'L')
    im.save(out_dir+os.sep +'std_train_inv.png','png')


def diffList(list1,list2):
    i=0
    res=[]
    for a,b in zip(list1,list2):
        if a!=b:
            res.append(i)
        i+=1
    return res

#methodeName - knn,svm,tree
#labelOfData - a text for folder name (in order to distinguish data, pca data, deskew  data etc)
#listMethodeParams - a list of disct (parameters for method are packed in dictionary)
def appllyClassify(methodName,labelOfData,trainData,trainLabels,testData,testLabels,listMethodParams):
    classyfier=None

    dirName = out_dir + os.sep + labelOfData + os.sep + methodName

    if not os.path.exists(dirName):
        os.makedirs(dirName)

    trainError = []
    testError =  []

    for param in listMethodParams:

        print(methodName,param,datetime.datetime.now())

        if methodName=='knn':
            classyfier = KNeighborsClassifier(**param)
        if methodName=='svm':
            classyfier = SVC(**param)
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

        trainError.append(len(diffTrain)/len(trainLabels))
        testError.append(len(diffTest)/len(testLabels))

        print(methodName,datetime.datetime.now(),'4')

    pickle.dump(listMethodParams,open(dirName + os.sep + 'parameters','wb'))
    pickle.dump(trainError,open(dirName + os.sep + 'trainerror','wb'))
    pickle.dump(testError,open(dirName + os.sep + 'testerror','wb'))

    print(trainError)
    print(testError)


#main

train_path = './train/'
test_path = './test/'

train_picture_file = 'train-images.idx3-ubyte'
train_label_file = 'train-labels.idx1-ubyte'

test_picture_file = 't10k-images.idx3-ubyte'
test_label_file  = 't10k-labels.idx1-ubyte'

out_dir='./out'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

train_pictures =  ~idx2numpy.convert_from_file(train_path+train_picture_file) #0 means black usually! And 255 means white!
train_labels = idx2numpy.convert_from_file(train_path+train_label_file)


test_pictures =  ~idx2numpy.convert_from_file(test_path+test_picture_file)
test_labels = idx2numpy.convert_from_file(test_path+test_label_file)

train_form = [train_pictures.shape[0],28*28]
test_from  = [test_pictures.shape[0],28*28]

train_pictures_as_feature_list = train_pictures.reshape(train_form)
test_pictures_as_feature_list  = test_pictures.reshape(test_from)


knn_params = [ {'n_neighbors':1}, {'n_neighbors':3}, {'n_neighbors':5}, {'n_neighbors':7}, {'n_neighbors':9}, {'n_neighbors':11}]


appllyClassify('knn','usual',train_pictures_as_feature_list[0:10000],train_labels[0:10000],test_pictures_as_feature_list,test_labels,knn_params)


for arg in sys.argv:
    if arg == 'step1':
        calcMeanAndStd()






