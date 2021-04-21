import sys
import matlab
import matlab.engine
import os
import shutil


def bse_ml():
    # step 1: nii.gz -> bmp
    subject_path = sys.argv[2]
    engine = matlab.engine.start_matlab()
    path = os.path.abspath(__file__)
    path_to_functions = path[:-11]
    engine.addpath(path_to_functions, nargout=0)
    engine.addpath(path_to_functions + '/NIfTI', nargout=0)
    n3 = engine.input_fn(subject_path)
    n3 = int(n3)
    engine.desktop(nargout=0)

    # step 2: create h5py
    command1 = 'python ' + path_to_functions + '/create_h5.py '
    os.system(command1)

    # step 3: predict
    command2 = 'python ' + path_to_functions + '/main.py ' + subject_path + ' ' + str(n3)
    os.system(command2)

    # step 4: save results
    if len(sys.argv) > 4:
        output_name = sys.argv[4]
        engine.save_pred(output_name, nargout=0)
        engine.desktop(nargout=0)
    else:
        engine.save_pred('meaningless', nargout=0)
        engine.desktop(nargout=0)

    # step 5: delete intermediate files
    shutil.rmtree('./pred_results/')
    shutil.rmtree('./pred_results_png/')
    shutil.rmtree('./test_data/')
