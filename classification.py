import scipy.io as sio # The library to deal with .mat

matstruct_squeezed = sio.loadmat('facialPoints.mat', squeeze_me=True)
points=matstruct_squeezed['points']

matstruct_squeezed = sio.loadmat('labels.mat', squeeze_me=True)
labels=matstruct_squeezed['labels']
