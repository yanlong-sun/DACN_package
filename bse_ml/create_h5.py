import skimage.io as io
import h5py
import glob
import sys

slices_path = glob.glob('./test_data/slices/*.bmp')
masks_path = glob.glob('./test_data/masks/*.bmp')
slices_list = io.ImageCollection(slices_path)
masks_list = io.ImageCollection(masks_path)
h5py_file_name = './test_data/test_data.hdf5'
f = h5py.File(h5py_file_name, 'w')
f.create_dataset('X', data=slices_list)
f.create_dataset('Y', data=masks_list)
f.close()

