import tensorflow as tf
import numpy as np
import matplotlib.pyplot
import cv2
import scipy.io as sio
import sys
import os
import math

from util import *
from visualization import *

X_in_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='X_in_image')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                for k in range(results.shape[2]):
                    f.write('{:.3f} '.format(results[i, j, k]))
            f.write('\n')

class multi_resnet():

    def __init__(self):
        pass

    def conv_op(self, x, name, n_out, training, useBN, kh=3, kw=3, dh=1, dw=1, padding="SAME", activation=tf.nn.relu):

        n_in = x.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
            z = tf.nn.bias_add(conv, b)
            if useBN:
                z = tf.layers.batch_normalization(z, trainable=training)
            if activation:
                z = activation(z)
            return z

    def max_pool_op(self, x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):

        return tf.nn.max_pool(x,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding=padding,
                              name=name)

    def fc_op(self, x, name, n_out, activation=tf.nn.relu):

        n_in = x.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.01))

            fc = tf.matmul(x, w) + b

            out = activation(fc)

        return fc, out

    def res_block_layers(self, x, name, n_out_list, change_dimension=False, block_stride=1):

        if change_dimension:
            short_cut_conv = self.conv_op(x, name + "_ShortcutConv", n_out_list[1], training=True, useBN=True, kh=1,
                                          kw=1,
                                          dh=block_stride, dw=block_stride,
                                          padding="SAME", activation=None)
        else:
            short_cut_conv = x

        block_conv_1 = self.conv_op(x, name + "_lovalConv1", n_out_list[0], training=True, useBN=True, kh=1, kw=1,
                                    dh=block_stride, dw=block_stride,
                                    padding="SAME", activation=tf.nn.relu)

        block_conv_2 = self.conv_op(block_conv_1, name + "_lovalConv2", n_out_list[0], training=True, useBN=True, kh=3,
                                    kw=3,
                                    dh=1, dw=1,
                                    padding="SAME", activation=tf.nn.relu)

        block_conv_3 = self.conv_op(block_conv_2, name + "_lovalConv3", n_out_list[1], training=True, useBN=True, kh=1,
                                    kw=1,
                                    dh=1, dw=1,
                                    padding="SAME", activation=None)

        block_res = tf.add(short_cut_conv, block_conv_3)
        res = tf.nn.relu(block_res)
        return res

    def bulid_resNet(self, x, training=True, usBN=True):

        conv1 = self.conv_op(x, "conv1", 32, training, usBN, 3, 3, 1, 1)
        # conv2 = self.conv_op(conv1, "conv2", 32, training, usBN, 3, 3, 1, 1)
        pool1 = self.max_pool_op(conv1, "pool1", kh=2, kw=2)
        # print("pool1.shape", pool1.shape)

        block1_1 = self.res_block_layers(pool1, "block2_1", [64, 256], True, 2)
        block1_2 = self.res_block_layers(block1_1, "block2_2", [64, 256], False, 1)
        block1_3 = self.res_block_layers(block1_2, "block2_3", [64, 256], False, 1)

        block2_1 = self.res_block_layers(block1_3, "block2_1", [128, 512], True, 2)
        block2_2 = self.res_block_layers(block2_1, "block2_2", [128, 512], False, 1)
        block2_3 = self.res_block_layers(block2_2, "block2_3", [128, 512], False, 1)
        block2_4 = self.res_block_layers(block2_3, "block2_4", [128, 512], False, 1)
        # print("block2_4", block2_4.shape)

        block3_1 = self.res_block_layers(block2_4, "block3_1", [256, 1024], True, 2)
        block3_2 = self.res_block_layers(block3_1, "block3_2", [256, 1024], False, 1)
        block3_3 = self.res_block_layers(block3_2, "block3_3", [256, 1024], False, 1)
        block3_4 = self.res_block_layers(block3_3, "block3_4", [256, 1024], False, 1)
        block3_5 = self.res_block_layers(block3_4, "block3_5", [256, 1024], False, 1)
        block3_6 = self.res_block_layers(block3_5, "block3_6", [256, 1024], False, 1)
        # print("block3_6.shape", block3_6)

        block4_1 = self.res_block_layers(block3_6, "block4_1", [256, 1024], True, 2)
        block4_2 = self.res_block_layers(block4_1, "block4_2", [256, 1024], False, 1)
        block4_3 = self.res_block_layers(block4_2, "block4_3", [256, 1024], False, 1)
        # print("block4_3.shape", block4_3)

        fc = tf.reshape(block4_3, (-1,4*4*1024))
        # print("fc.shape", fc.shape)

        fc1, _ = self.fc_op(fc, "fc1", 1024)
        fc1 = tf.layers.dropout(fc1, keep_prob)

        fc2, _ = self.fc_op(fc1, "fc2", 1024)
        fc2 = tf.layers.dropout(fc2, keep_prob)

        fc3, _ = self.fc_op(fc2, "fc3", 42)


        return fc3



pred = multi_resnet().bulid_resNet(X_in_image)

"""
========================================
Test the model
========================================
"""

"""
load the test sets 
"""
# path = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/6.Multi-ResNet/results/'
# dir = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/6.Multi-ResNet/data/nyu_dataset/'
# image_test = np.load(dir + 'NYU_Image_Test.npy').reshape([-1, 128, 128, 1])
# label_test = np.load(dir + 'NYU_Label_Test.npy').reshape([-1, 42])
#
# test_dir = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/6.Multi-ResNet/data/nyu_dataset/'
# labels = sio.loadmat(test_dir + "joint_data.mat")
# joint_uvd = labels['joint_uvd'][0]  # shape: (8252,36,3)
# joint_xyz = labels['joint_xyz'][0]  # shape: (8252,36,3)
# joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])
# com = np.load(test_dir+"Com_test.npy")

dir_image_test = 'H:/HandPoseEstimation/dataset/NYU/test'

test_image = np.zeros(((8252, 480, 640)))
for id in range(0, 8252):

    img_path = '{}/depth_1_{:07d}.png'.format(dir_image_test, id + 1)  # NYU/train/depth_1_{:07d}.png

    if not os.path.exists(img_path):
        print('{} Not Exists!'.format(img_path))
        continue

    img = cv2.imread(img_path) # shape:(480,640,3)
    depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256) # shape:(480,640)
    test_image[id] = depth

np.save('NYU_Image_Test.npy',test_image)
exit()

# """
# # calculate the average error and maxiunum error
# """
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#
#     # sess.run(init)
#     saver.restore(sess, "/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/6.Multi-ResNet/model/nyu/model_99.ckpt")
#
#     # obtain the ground-truth joints
#     labels_norm = np.zeros(((8252, 14, 3)))
#     for i, labels_gt in enumerate(label_test):
#         labels_norm[i] = labels_gt.reshape([14,3])  + joint_xyz[i,34]
#
#     # obtain the joints that network outputs
#     outputs = np.zeros((8252,42))
#     for i in range(0, label_test.shape[0]//50):
#         image = image_test[i*50:(i+1)*50] / 150.0
#         outputs[i*50:(i+1)*50] = sess.run(pred, feed_dict={X_in_image: image, keep_prob: 1})
#
#     outputs_labels = np.zeros(((8252, 14, 3)))
#     for i in range(0, 8252):
#         outputs_labels[i] = outputs[i].reshape([14,3]) * 150.  + joint_xyz[i,34]
#
#     assert labels_norm.shape == outputs_labels.shape
#
#     # calculate the average error, max error between the ground-truth and the prediction joints
#     average_error = np.nanmean(np.nanmean(np.sqrt(np.square(labels_norm - outputs_labels).sum(axis=2)), axis=1))
#     max_error = np.nanmax(np.sqrt(np.square(labels_norm - outputs_labels).sum(axis=2)))
#
#     print("average_error: %f mm"%(average_error))
#     print("max_error: %f mm"%(max_error))
#
#     dir = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/6.Multi-ResNet/results/nyu/'
#     out_file = dir + 'xzz.txt'
#     outputs_labels_uvd = world2pixel(outputs_labels, 588.036865, 587.075073, 320, 240)
#
#     save_results(outputs_labels_uvd, out_file)
#
#     image_test = np.load('nyu_Image_Test.npy')
#
#     ###save the ground truth
#     for i in range(len(image_test)):
#         if i%200 == 0:
#             save_figure_joint_skeleton(image_test[i].reshape([480, 640]),
#                                        outputs_labels_uvd[i].reshape([14,3]),
#                                        joint_uvd, i, 1)
#
#     ###save the prediction
#     for i in range(len(image_test)):
#         if i%200 == 0:
#             save_figure_joint_skeleton(image_test[i].reshape([480, 640]),
#                                        outputs_labels_uvd[i].reshape([14,3]),
#                                        joint_uvd, i, 0)
#
#     ###contrast ground truth with predicton
#     for i in range(len(image_test)):
#         if i%200 == 0:
#             save_figure_joint_skeleton(image_test[i].reshape([480, 640]),
#                                        outputs_labels_uvd[i].reshape([14,3]),
#                                        joint_uvd, i, 2)
#
#
#
#
#
#
#
#


