import sys
import matlab
import matlab.engine
import os
import shutil


# step 1: nii.gz -> bmp
subjects_folder_path = sys.argv[1]
engine = matlab.engine.start_matlab()
path = os.path.abspath(__file__)
path_to_functions = path[:-7]
engine.addpath(path_to_functions, nargout=0)
engine.addpath(path_to_functions + '/NIfTI', nargout=0)
n3 = engine.input_fn(subjects_folder_path)
n3 = int(n3)
engine.desktop(nargout=0)

# step 2: create h5py
command1 = 'python ' + path_to_functions + '/create_h5.py ' + subjects_folder_path
os.system(command1)

# step 3: predict
command2 = 'python ' + path_to_functions + '/main.py ' + subjects_folder_path + ' ' + str(n3)
os.system(command2)

# step 4: save results
engine.save_pred(subjects_folder_path, nargout=0)
engine.desktop(nargout=0)

# step 5: delete intermediate files
shutil.rmtree(subjects_folder_path + 'pred_results/')
shutil.rmtree(subjects_folder_path + 'pred_results_png/')
shutil.rmtree(subjects_folder_path + 'test_data/')