import numpy as np
import os
import pickle
from itertools import groupby 
import sys
import h5py
import pandas as pd
import nrrd
from skimage import io
import matplotlib.pyplot as plt
plt.switch_backend('agg')




def open_Beh_Jpos_GC_DicData(pathDic, filename):

    print('Opening',  filename ,'...')

    if os.path.exists(pathDic+'/'+filename):

        Beh_Jpos_GC_DicData = pickle.load(open( pathDic+'/'+filename, "rb" ))


        return Beh_Jpos_GC_DicData

    else:

        print('data not found...')
        sys.exit(0)

        return



