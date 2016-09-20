__author__ = 'm.baytel'

import idx2numpy
from PIL import Image
import numpy
import os
import common


def calcMeanAndStd(pictureFile,labelFile,out_dir):
    pictures =  ~idx2numpy.convert_from_file(pictureFile) #0 means black usually! And 255 means white!
    labels    = idx2numpy.convert_from_file(labelFile)

    mean_picture = numpy.mean(pictures,0)
    std_picture  = numpy.std(pictures,0)

    pictures_by_classes = [[] for i in range(10)]
    pictures_by_classes_array = [None for i in range(10)]

    for picture,label in zip(pictures,labels):
        pictures_by_classes[label].append(picture)


    for i in range(10):
        pictures_by_classes_array[i] = numpy.array(pictures_by_classes[i])

        mean_image = Image.fromarray(numpy.mean(pictures_by_classes_array[i],axis=0).astype('uint8'),'L')
        std_image = Image.fromarray(numpy.std(pictures_by_classes_array[i],axis=0).astype('uint8'),'L')
        std_image_inv = Image.fromarray(~numpy.std(pictures_by_classes_array[i],axis=0).astype('uint8'),'L')

        mean_image.save(out_dir+os.sep + 'mean_train_' + str(i)+'.png','png')
        std_image.save(out_dir+os.sep + 'std_train_' + str(i)+'.png','png')
        std_image_inv.save(out_dir+os.sep + 'std_train_inv_' + str(i)+'.png','png')


    im = Image.fromarray(mean_picture.astype('uint8'),'L')
    im.save(out_dir+os.sep +'mean_train.png','png')

    im = Image.fromarray(std_picture.astype('uint8'),'L')
    im.save(out_dir+os.sep +'std_train.png','png')

    im = Image.fromarray(~std_picture.astype('uint8'),'L')
    im.save(out_dir+os.sep +'std_train_inv.png','png')


calcMeanAndStd(common.train_images,common.train_labels,common.out_dir)















