import h5py
import os
import numpy as np
import scipy.io as sio


def loadPose(dir, limit):
    poseData = []
    counter = 0
    for file in os.listdir('data/BP4D/' + dir):
        if file.endswith(".mat"):
            # Initializing h5py file handler
            my_path = os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(my_path, "data/BP4D/2DFeatures", file)
            fh = h5py.File(filepath, 'r')
            # Extracting the list of landmarks array objects from pred field
            lms_obj = fh.get('fit/pose')
            # # Initializing output 3d array
            all_lms_array = np.zeros((len(lms_obj), 3, 3))
            # Iterate over the list to fetch each frame’s landmarks array
            for i in range(0, len(lms_obj)):
                rot = fh[lms_obj[i][0]].get('rot')
                # Returns 3*3 numpy array
                all_lms_array[i] = rot.value
            poseData.append(all_lms_array)

        counter = counter + 1
        if (counter == limit):
            break

    return poseData


Y = loadPose('2DFeatures', 10)
print(Y)
