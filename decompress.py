__author__ = 'm.baytel'

import common
from sklearn.decomposition import PCA
import math
import idx2numpy
import numpy
import cv2
from PIL import Image
import os



def deskew(picture,outname):

    res=numpy.transpose((picture.astype('uint8')<255).nonzero())
    pca = PCA(n_components=2)
    pca.fit(res)

    #taking of axis as X is heuristic
    tan1=pca.components_[0][1]/pca.components_[0][0]

    res=tan1

    angle=math.atan(res)*180/math.pi

    rows,cols = picture.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle,1)
    dst = cv2.warpAffine(picture,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=[255,255,255])

    cv2.imwrite(outname,dst)

    return ~dst


def decompressAndDeskew(picture_idx_name,label_idx_name,out_dir,deskew_dir,deskew_idx):
    pictures =  ~idx2numpy.convert_from_file(picture_idx_name) #0 means black usually! And 255 means white!
    labels   =   idx2numpy.convert_from_file(label_idx_name)

    for i in range(10):
        name=out_dir + os.sep +str(i)
        deskewName=deskew_dir + os.sep +str(i)

        if not os.path.exists(name):
            os.makedirs(name)

        if not os.path.exists(deskewName):
            os.makedirs(deskewName)

    last_number=[0]*10

    deskewList=[]

    for picture,label in zip(pictures,labels):
        number = last_number[label]
        last_number[label] = number+1
        im = Image.fromarray(picture,'L')
        im.save(out_dir + os.sep +str(label)+os.sep+str(number)+'.png','png')

        deskewList.append(deskew(picture,deskew_dir + os.sep +str(label)+os.sep+str(number)+'.png'))

    f_write = open(deskew_idx, 'wb')
    idx2numpy.convert_to_file(f_write, numpy.array(deskewList))


print('Decompress test data...')

decompressAndDeskew(common.test_images,common.test_labels,common.internal_test_dir,common.deskew_test_dir,common.test_deskew_images)


print('Decompress train data...')

decompressAndDeskew(common.train_images,common.train_labels,common.internal_train_dir,common.deskew_train_dir,common.train_deskew_images)
