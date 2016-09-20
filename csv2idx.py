__author__ = 'm.baytel'

import os
import common
import idx2numpy
import csv
import numpy
#from PIL import Image

def csv2idx(csvpath):
    with open(csvpath, newline='') as csvfile:
        picturereader = csv.reader(csvfile)

        first=True
        withLabels=False

        labelList   = []
        pictureList = []

        filename = os.path.basename(csvpath)

        filename = filename.split('.')[0]

        picname = filename + '-pic.idx3-ubyte'
        labname = filename + '-lab.idx3-ubyte'

        for row in picturereader:
            if first:
                first=False
                withLabels = (len(row) == 28*28+1)
                continue

            if withLabels:
                labelList.append(row[0])
                row = row[1:]

            picArray = numpy.array(row)
            picArray = picArray.astype('uint8')
            pictureList.append(picArray)

            #im = Image.fromarray(~picArray.reshape(28,28),'L')
            #im.save('XXX.png')

    idx2numpy.convert_to_file(open(picname,'wb'),numpy.array(pictureList))

    if withLabels:
        idx2numpy.convert_to_file(open(labname,'wb'),numpy.array(labelList).astype('uint8'))






csv2idx(common.data_dir + os.sep + 'train.csv')

csv2idx(common.data_dir + os.sep + 'test.csv')
