import h5py
import os
import numpy as np


def loadPred(dir, limit):
    predData = []
    counter = 0
    for file in os.listdir('data/BP4D/' + dir):
        if file.endswith(".mat"):
            # Initializing h5py file handler
            my_path = os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(my_path, "data/BP4D/2DFeatures", file)
            fh = h5py.File(filepath, 'r')
            # Extracting the list of landmarks array objects from pred field
            lms_obj = fh.get('fit/pred')
            # Initializing output 3d array
            all_lms_array = np.zeros((len(lms_obj), 49, 2))
            # Iterate over the list to fetch each frame’s landmarks array
            for i in range(0, len(lms_obj)):
                # Returns 49*2 numpy array
                all_lms_array[i] = fh[lms_obj[i][0]].value.transpose()
            predData.append(all_lms_array)

        counter = counter + 1
        if (counter == limit):
            break

    return predData


def loadPose(dir, limit):
    predData = []
    counter = 0
    for file in os.listdir('data/BP4D/' + dir):
        if file.endswith(".mat"):
            # Initializing h5py file handler
            my_path = os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(my_path, "data/BP4D/2DFeatures", file)
            fh = h5py.File(filepath, 'r')
            # Extracting the list of landmarks array objects from pred field
            lms_obj = fh.get('fit/pose/rot')
            # Initializing output 3d array
            all_lms_array = np.zeros((len(lms_obj), 3, 3))
            # Iterate over the list to fetch each frame’s landmarks array
            for i in range(0, len(lms_obj)):
                # Returns 49*2 numpy array
                all_lms_array[i] = fh[lms_obj[i][0]].value
            predData.append(all_lms_array)

        counter = counter + 1
        if (counter == limit):
            break

    return predData


X = loadPred('2DFeatures', 10)
Y = loadPose('2DFeatures', 10)
print(Y)
