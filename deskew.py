__author__ = 'm.baytel'

import os
from sklearn.decomposition import PCA
import math
import idx2numpy
import numpy
import cv2
from PIL import Image

in_dir_train  = '.\\internal\\train'
in_dir_test  = '.\\internal\\test'

out_dir = '.\\deskew'

train_path = 'train\\'
test_path = 'test\\'


out_train = out_dir + os.sep + train_path
out_test = out_dir + os.sep + test_path

out_train_idx = train_path + 'train-deskew.idx3-ubyte'
out_test_idx  = test_path + 'test-deskew.idx3-ubyte'

if not os.path.exists(out_train):
    os.makedirs(out_train)

if not os.path.exists(out_test):
    os.makedirs(out_test)


def PIL2array(img):
    tmp=numpy.array(img.getdata(),numpy.uint8)
    tmp = tmp.reshape(img.size[1], img.size[0])
    return tmp

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = numpy.c_[arr, 255*numpy.ones((len(arr),1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


def skewAngle(filename,outname):
    img = Image.open(filename)
    picture = PIL2array(img)

    res=numpy.transpose((picture.astype('uint8')<255).nonzero())
    pca = PCA(n_components=2)
    pca.fit(res)

    #taking of axis as X is heuristic
    tan1=pca.components_[0][1]/pca.components_[0][0]

    #tan2=pca.components_[1][1]/pca.components_[1][0]

    res=tan1

    #print(tan1,tan2)

    #if (abs(tan2) < abs(tan1)):
        #res=tan2

    #print(pca.components_)


    angle=math.atan(res)*180/math.pi

    #print('angle',math.atan(tan1)*180/math.pi )
    #print('alt angle',math.atan(tan2)*180/math.pi )

    imgCV=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

    #print(imgCV.shape)

    rows,cols = imgCV.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle,1)
    dst = cv2.warpAffine(imgCV,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=[255,255,255])

    cv2.imwrite(outname,dst)

    img = Image.open(outname)
    return ~PIL2array(img)


def applyDeskew(inputDir,outPutDir,listOfPicture):
    for path in os.listdir(inputDir):
        #print(path)
        outPath= outPutDir + os.sep + path
        path = inputDir+os.sep+path
        if os.path.isdir(path):
            if not os.path.exists(outPath):
                 os.makedirs(outPath)
            applyDeskew(path,outPath,listOfPicture)
        else:
            listOfPicture.append(skewAngle(path,outPath))

    return listOfPicture


print('Deskew train data...')

res=applyDeskew(in_dir_train ,out_train,[])
print("Train len=",len(res))
f_write = open(out_train_idx, 'wb')
idx2numpy.convert_to_file(f_write, numpy.array(res))

print('Deskew test data...')

res=applyDeskew(in_dir_test ,out_test,[])
print("Test len=",len(res))
f_write = open(out_test_idx, 'wb')
idx2numpy.convert_to_file(f_write, numpy.array(res))


