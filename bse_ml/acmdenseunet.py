import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from img_utils import imsave
import ops

class AcmDenseUnet(object):

    def __init__(self, sess, conf, is_train):
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.is_train = is_train
      
        self.data_format = 'NHWC'
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [conf.batch, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]     
        
        self.use_new = True
        
#—————————————————————————————————————————————————————# 
  
    def inference(self, inputs):
        #——————————————  step：1  ——————————————#——start-block———#
        rate_field = 0  # Useless parameters
        outputs = inputs
       
        name = 'start_block'
        outputs = self.down_conv_func()(outputs, rate_field, 96, (3, 3), name+'/conv1', stride=2, is_train=self.is_train, norm=False)
        outputs = self.down_conv_func()(outputs, rate_field, 96, (3, 3), name+'/conv2', is_train=self.is_train, norm=False)
        conv1 = self.down_conv_func()(outputs, rate_field, 96, (3, 3), name+'/conv3', is_train=self.is_train, norm=False)
      
        outputs = ops._max_pool2d(conv1, (3, 3), name+'/max_pool')
        
        #——————————————  step：2  ——————————————#——downsampling———#
        
        # 1
        name = 'dense_block1'
        block1 = self.dense_block(outputs, name+'/dense', 6)
        outputs = ops.conv2d(block1, rate_field, 192, (1,1), name+'/conv11', is_train=self.is_train, bias=False)
        outputs = ops._avg_pool2d(outputs, (3, 3), name)
        
        # 2
        name = 'dense_block2'
        block2 = self.dense_block(outputs, name+'/dense', 12)
        outputs = ops.conv2d(block2, rate_field, 384, (1,1), name+'/conv11', is_train=self.is_train, bias=False)
        outputs = ops._avg_pool2d(outputs, (3, 3), name)

        
        # 3
        name = 'dense_block3'
        block3 = self.dense_block(outputs, name+'/dense', 36)
        outputs = ops.conv2d(block3, rate_field, 1056, (1,1), name+'/conv11', is_train=self.is_train, bias=False)
        outputs = ops._avg_pool2d(outputs, (3, 3), name)

        
        # 4
        name = 'dense_block4'
        block4 = self.dense_block(outputs, name+'/dense', 24)
        block4 = ops.conv2d(block4, rate_field, 2112, (1,1), name+'/conv11', is_train=self.is_train, bias=False)
        
        #——————————————  step：3  ——————————————#——upsampling———#
        
        # 1
        name = 'up1'
        h = 2*outputs.shape[1]
        w = 2*outputs.shape[2]
        outputs = tf.image.resize_bilinear(block4, size=(h,w), align_corners=True, name=name+'/bilinear')
        outputs = outputs+block3
        h = 2*outputs.shape[1]
        w = 2*outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 768, self.conv_size, name+'/conv33', is_train=self.is_train, bias=False)
        
        # 2
        name = 'up2'
        
        outputs = tf.image.resize_bilinear(outputs, size=(h,w), align_corners=True, name=name+'/bilinear')
        outputs = outputs+block2
        h = 2*outputs.shape[1]
        w = 2*outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 384, self.conv_size, name+'/conv33', is_train=self.is_train, bias=False)
        
        # 3
        name = 'up3'
        
        outputs = tf.image.resize_bilinear(outputs, size=(h,w), align_corners=True, name=name+'/bilinear')
        outputs = outputs+block1
        h = 2*outputs.shape[1]
        w = 2*outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 96, self.conv_size, name+'/conv33', is_train=self.is_train, bias=False)
        
        # 4
        name = 'up4'
       
        outputs = tf.image.resize_bilinear(outputs, size=(h,w), align_corners=True, name=name+'/bilinear')
        outputs = outputs+conv1
        h = 2*outputs.shape[1]
        w = 2*outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 96, self.conv_size, name+'/conv33', is_train=self.is_train, bias=False)
        
        # 5
        name = 'up5'
     
        outputs = tf.image.resize_bilinear(outputs, size=(h,w), align_corners=True, name=name+'/bilinear')
        
        if self.use_new:
            
            #----------branch 1-----
            branch1 = ops.conv2d(outputs, rate_field, 32, (3,3), name+'/branch1-1', is_train=self.is_train, bias=False)
            branch1 = ops.conv2d(branch1, rate_field, 32, (3,3), name+'/branch1-2', is_train=self.is_train, bias=False)
            branch1 = ops.conv2d(branch1, rate_field, 1, (1,1), name+'/branch1-3', is_train=self.is_train, bias=False)
            
            #----------branch 2-----
            branch2 = ops.conv2d(outputs, rate_field, 32, (3,3), name+'/branch2-1', is_train=self.is_train, bias=False)
            branch2 = ops.conv2d(branch2, rate_field, 32, (3,3), name+'/branch2-2', is_train=self.is_train, bias=False)
            branch2 = ops.conv2d(branch2, rate_field, 1, (1,1), name+'/branch2-3', is_train=self.is_train, bias=False)
            
            #----------branch 3-----
            branch3 = ops.conv2d(outputs, rate_field, 32, (3,3), name+'/branch3-1', is_train=self.is_train, bias=False)
            branch3 = ops.conv2d(branch3, rate_field, 32, (3,3), name+'/branch3-2', is_train=self.is_train, bias=False)
            branch3 = ops.conv2d(branch3, rate_field, 2, (1,1), name+'/branch3-3', is_train=self.is_train, bias=False)
            
            outputs = tf.concat([branch1, branch2, branch3], self.channel_axis, name=name+'/concat2')
            
        else:
            outputs = ops.conv2d(outputs, rate_field, 64, self.conv_size, name+'/conv33', is_train=self.is_train, bias=False)
            outputs = ops.conv2d(outputs, rate_field, 4, (1,1), name+'/conv11', is_train=self.is_train, bias=False)

        return outputs,rate_field
#————————————————————————————dense-block —————————————————————————# 
    
    def dense_block(self, inputs, name, num):
        rate_field = inputs
       
        for i in range(num):
            
            outputs = ops.conv2d(inputs, rate_field, 192, (1,1), name+'/conv11_'+str(i+1), is_train=self.is_train, bias=False)
            outputs = ops.conv2d(outputs, rate_field, 48, self.conv_size, name+'/conv33_'+str(i+1), is_train=self.is_train, bias=False)
            
            inputs = tf.concat([inputs, outputs], self.channel_axis, name=name+'/concat'+str(i+1))
       
        return inputs
#————————————————————————————————————————————————————# 

    def down_conv_func(self):
        return getattr(ops, self.conf.down_conv_name)
    
    def bottom_conv_func(self):
        return getattr(ops, self.conf.bottom_conv_name)
    
    def up_conv_func(self):
        return getattr(ops, self.conf.up_conv_name)
    
    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)