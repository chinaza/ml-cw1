import h5py
import os
import numpy as np
import scipy.io as sio


def loadPose(dir, limit):
    predData = []
    counter = 0
    for file in os.listdir('data/BP4D/' + dir):
        print(file)
        if file.endswith(".mat"):
            # Initializing h5py file handler
            my_path = os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(my_path, "data/BP4D/2DFeatures", file)
            fh = h5py.File(filepath, 'r')
            # Extracting the list of landmarks array objects from pred field
            lms_obj = fh.get('fit/pose')
            print(lms_obj[0])
        break


Y = loadPose('2DFeatures', 10)
