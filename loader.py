import h5py
import os
import numpy as np


def loadData(dir):
    for file in os.listdir('BP4D/' + dir):
        if file.endswith(".mat"):
            print(file)
            # fh = h5py.File(file, 'r')  # Initializing h5py file handler
            # # Extracting the list of landmarks array objects from pred field
            # lms_obj = fh.get('fit/pred')
            # # Initializing output 3d array
            # all_lms_array = np.zeros((len(lms_obj), 49, 2))
            # # Iterate over the list to fetch each frame’s landmarks array
            # for i in range(0, len(lms_obj)):
            #     # Returns 49*2 numpy array
            #     all_lms_array[i] = fh[lms_obj[i][0]].value.transpose()


loadData('2DFeatures')
