import os
import time
import argparse
import numpy as np
import tensorflow as tf
from actions import Actions
import sys
def configure():

    flags = tf.app.flags
    
    #————————————————————————————--—————————————————————————# 
    flags.DEFINE_string('network_name', 'acmdenseunet', 'Use which framework:  unet, denseunet, deeplabv3plus')
    
    flags.DEFINE_integer('max_epoch', 100001, '# of step in an epoch')  # 100001
    flags.DEFINE_integer('test_step', 1000, '# of step to test a model')
    flags.DEFINE_integer('save_step', 1000, '# of step to save a model')
    
    flags.DEFINE_integer('valid_start_epoch', 1,'start step to test a model')
    flags.DEFINE_integer('valid_end_epoch', 100001, 'end step to test a model')
    flags.DEFINE_integer('valid_stride_of_epoch',1000, 'stride to test a model')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
    flags.DEFINE_integer('test_epoch', 99001, 'Test or predict epoch')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    
    flags.DEFINE_integer('summary_step', 10000000, '# of step to save the summary')
    #—————————————————————————————————————————————————————# 
    
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('beta1', 0.9, 'beta1')
    flags.DEFINE_float('beta2', 0.99, 'beta2')
    flags.DEFINE_float('epsilon', 1e-8, 'epsilon')
 
    flags.DEFINE_integer('gpu_num', 1, 'the number of GPU')
    #—————————————————————————————————————————————————————#
    subjects_name = sys.argv[1]
    subjects_name = subjects_name[:-7]
    slice_num = int(sys.argv[2])
    for i in range(6, 1, -1):
        if slice_num % i == 0:
            batch_size = i
            break
        else:
            batch_size = 1

    print('slice_num', slice_num)
    print('batch_size', batch_size)
    flags.DEFINE_string('data_dir', './test_data/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'training_data.hdf5', 'Training data')
    flags.DEFINE_string('valid_data', 'valid_data.hdf5', 'Validation data')
    flags.DEFINE_string('test_data', 'test_data.hdf5', 'Testing data')
    flags.DEFINE_integer('valid_num', 3840,'the number of images in the validing set')
    flags.DEFINE_integer('test_num', slice_num,'the number of images in the testing set')  # add test data bs:2907, cc: 4800
    flags.DEFINE_integer('batch', batch_size, 'batch size')              # 4
    flags.DEFINE_integer('batchsize', batch_size, 'total batch size')     # 4
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 256, 'height size')
    flags.DEFINE_integer('width', 256, 'width size')
    flags.DEFINE_boolean('is_training', True, '是否训练') 
    flags.DEFINE_integer('class_num', 2, 'output class number')
    #————————————————————————————-—————————————————————————#
    flags.DEFINE_string('network_dir', '../network4/', 'network_dir')
    flags.DEFINE_string('logdir', '../network4/logdir/', 'Log dir')
    path = os.path.abspath(__file__)
    path_to_bse_ml = path[:-7]
    flags.DEFINE_string('modeldir', path_to_bse_ml + '/modeldir/', 'Model dir')
    flags.DEFINE_string('sample_dir', './test_data/', 'Sample directory')
    flags.DEFINE_string('sample_net_dir', './pred_results_png/', 'Sample directory')
    flags.DEFINE_string('record_dir', '../network4/record/', 'Experiment record directory')
    #————————————————————————————-—————————————————————————# 
    flags.DEFINE_boolean('use_asc', False, 'use ASC or not')
    flags.DEFINE_string('down_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d, adaptive_separate_conv2d')
    flags.DEFINE_string('bottom_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d')
    flags.DEFINE_string('up_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d')
   
    flags.DEFINE_string('deconv_name', 'deconv', 'Use which deconv op: deconv')
      
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def predict():
    predict_loss = []
    predict_accuracy = []
    predict_dice = []
    model = Actions(sess, configure())
    loss, acc, dice = model.predict()
    predict_loss.append(loss)
    predict_accuracy.append(acc)
    predict_dice.append(dice)


# ———————————————————————————— main —————————————————————————#
"""
函数功能：主函数，设置不同的action
"""


def main(argv):
    predict()


# ———————————————————————————— GPU设置 —————————————————————————#
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.app.run()
