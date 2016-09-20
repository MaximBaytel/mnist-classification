__author__ = 'm.baytel'

import matplotlib.pyplot as plt
import pickle
import common
import matplotlib.patches as mpatches


#def plotEigens(dumpFile,title):

ratiolist1 = pickle.load(open(common.out_ratio_usual,'rb'))
ratiolist2 = pickle.load(open(common.out_ratio_deskew,'rb'))
ax = plt.gca()
ax.grid(True)
green_patch  = mpatches.Patch(color='green', label='original data')
blue_patch = mpatches.Patch(color='blue', label='deskew data')
plt.legend(handles=[green_patch,blue_patch])
plt.plot(ratiolist1)
plt.plot(ratiolist2)
plt.title('test')
plt.show()


#plotEigens(common.out_ratio_usual,'aaa')

#plotEigens(common.out_ratio_deskew,'bb')
