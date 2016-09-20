__author__ = 'm.baytel'

import os

data_dir = './data'
out_dir = './out'
internal_dir = './internal'
deskew_dir = './deskew'

test_dir =  data_dir + os.sep + 'test'
train_dir =  data_dir + os.sep + 'train'

deskew_test_dir  =  deskew_dir + os.sep + 'test'
deskew_train_dir =  deskew_dir + os.sep + 'train'

internal_test_dir  = internal_dir + os.sep + 'test'
internal_train_dir = internal_dir + os.sep + 'train'


test_images  =  test_dir + os.sep + 't10k-images.idx3-ubyte'
test_images_pca  =  test_dir + os.sep + 'pca-images.idx3-ubyte'
test_deskew_images  =  test_dir + os.sep + 'deskew-images.idx3-ubyte'
test_deskew_pca_images  =  test_dir + os.sep + 'deskew-pca-images.idx3-ubyte'


train_images =  train_dir + os.sep + 'train-images.idx3-ubyte'
train_images_pca  =  train_dir + os.sep + 'pca-images.idx3-ubyte'
train_deskew_images  =  train_dir + os.sep + 'deskew-images.idx3-ubyte'
train_deskew_pca_images  =  train_dir + os.sep + 'deskew-pca-images.idx3-ubyte'

test_labels  =  test_dir + os.sep + 't10k-labels.idx1-ubyte'
train_labels =  train_dir + os.sep + 'train-labels.idx1-ubyte'

eigenPCA_dir= out_dir + os.sep + 'eigenPCA'
eigenPCA_deskewdir= out_dir + os.sep + 'eigenPCAdeskew'

out_ratio_usual  = out_dir + os.sep + 'ratio_usual'
out_ratio_deskew = out_dir + os.sep + 'ratio_deskew'


dirs = [out_dir,internal_dir,deskew_test_dir,deskew_train_dir,internal_test_dir,internal_train_dir,eigenPCA_dir,eigenPCA_deskewdir]

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)







