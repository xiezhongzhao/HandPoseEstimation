#keep compatability among different python version
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import scipy.io as sio
import cv2
import matplotlib.pyplot
import random
from util import *
from nyu_preprocessing import translationHand, scaleHand, rotateHand, joint3DToImg, jointImgTo3D, figure_joint_skeleton

#Define our input and output data
#load all augmented depth images and labels
dir = 'H:/HandPoseEstimation/dataset/NYU/'


#calculate training time
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"

#save array to txt file
def save_results(results, out_file):

    with open(out_file, 'w') as f:
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                for k in range(results.shape[2]):
                    f.write('{:.3f} '.format(results[i, j, k]))
            f.write('\n')


"""
========================================
Test the model
========================================
"""
def test_model(dir, epoch):
    """
    load the test sets
    """
    image_test = np.load(dir + 'NYU_Image_Test.npy').reshape([-1, 128, 128, 1])
    label_test = np.load(dir + 'NYU_Label_Test.npy').reshape([-1, 42])

    test_dir = os.path.join(dir, 'test/')
    labels_test = sio.loadmat(test_dir + "joint_data.mat")
    joint_uvd_test = labels_test['joint_uvd'][0]  # shape: (8252,36,3)
    joint_xyz_test = labels_test['joint_xyz'][0]  # shape: (8252,36,3)
    joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])


    """
    # calculate the average error and maxiunum error
    """
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, "H:/HandPoseEstimation/model/nyu/model_{}.ckpt".format(epoch))

        # obtain the ground-truth joints
        labels_norm = np.zeros(((8252, 14, 3)))
        for i, labels_gt in enumerate(label_test):
            labels_norm[i] = labels_gt.reshape([14,3]) * 150 + joint_xyz_test[i,34]

        # obtain the joints that network outputs
        outputs = np.zeros((8252,42))
        for i in range(0, label_test.shape[0]//50):
            image = image_test[i*50:(i+1)*50]
            outputs[i*50:(i+1)*50] = sess.run(pred, feed_dict={X_in_image: image, keep_prob: 1})

        outputs_labels = np.zeros(((8252, 14, 3)))
        for i in range(0, 8252):
            outputs_labels[i] = outputs[i].reshape([14,3]) * 150. + joint_xyz_test[i,34]

        assert labels_norm.shape == outputs_labels.shape

        # calculate the average error, max error between the ground-truth and the prediction joints
        average_error = np.nanmean(np.nanmean(np.sqrt(np.square(labels_norm - outputs_labels).sum(axis=2)), axis=1))
        max_error = np.nanmax(np.sqrt(np.square(labels_norm - outputs_labels).sum(axis=2)))

        print("average_error: %f mm"%(average_error))
        print("max_error: %f mm\n"%(max_error))

        # save the uvd prediction joints
        dir = 'H:/HandPoseEstimation/result/nyu/'
        out_file = os.path.join(dir, "epoch_%d_%.2fmm.txt" % (epoch, int(average_error)))
        outputs_labels_uvd = world2pixel(outputs_labels, 588.036865, 587.075073, 320, 240)

        save_results(outputs_labels_uvd, out_file)

    return average_error


X_in_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='X_in_image')
X_in_label = tf.placeholder(dtype=tf.float32, shape=[None, 42], name='X_in_label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

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

        #fc = tf.reshape(block4_3, (-1,4*4*1024))
        fc = tf.layers.flatten(block4_3)
        # print("fc.shape", fc.shape)

        fc1, _ = self.fc_op(fc, "fc1", 1024)
        fc1 = tf.layers.dropout(fc1, keep_prob)

        fc2, _ = self.fc_op(fc1, "fc2", 1024)
        fc2 = tf.layers.dropout(fc2, keep_prob)

        fc3, _ = self.fc_op(fc2, "fc3", 42)


        return fc3



"""
========================================
Train the model
========================================
"""
cube_size = 300
steps = []
dis_loss_list = []


path = 'H:/HandPoseEstimation/result/'

import random

dataset_path_train = 'H:/HandPoseEstimation/dataset/NYU/train/'
label_path_train = 'H:/HandPoseEstimation/dataset/NYU/train/joint_data.mat'
epoches = 1
batch_size = 64
learning_rate = 0.0001
average_error_flag = 30 # 30 mm

# pred = multi_resnet().concat_multi_scale(X_in_image)
pred = multi_resnet().bulid_resNet(X_in_image)
loss_joints =  tf.reduce_mean(tf.reduce_sum(tf.squared_difference(pred, X_in_label), 1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_joints)


labels = sio.loadmat(label_path_train)
joint_uvd = labels['joint_uvd'][0]  # shape: (72757,36,3)
joint_xyz = labels['joint_xyz'][0]  # shape: (72757,36,3)
joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])


'''
Start session and initialize all the variables
'''

init = tf.global_variables_initializer()
saver = tf.train.Saver()
start_time_sum = time.time()

with tf.Session() as sess:

    sess.run(init)

    # epoch iteration
    for epoch in range(epoches):

        idx = list(random.sample(range(0, 72757), 72757))
        rng = np.random.RandomState(23455)

        # batch at every epoch
        # for num_batch in range(0, len(idx) // batch_size):
        for num_batch in range(0, 1):

            batch_image_train = []
            batch_label_train = []

            # get batch images and poses
            for id in idx[batch_size*num_batch : batch_size*(num_batch+1)]:
                # rand original depth image
                img_path = '{}/depth_1_{:07d}.png'.format(dataset_path_train, id + 1)  # NYU/train/depth_1_{:07d}.png

                if not os.path.exists(img_path):
                    print('{} Not Exists!'.format(img_path))
                    continue
                img = cv2.imread(img_path)  # shape:(480,640,3)
                ori_depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256)  # shape: (480,640)

                # random factor
                sigma_sc = 0.02
                sigma_com = 10.
                num_poses = 1.0
                sc = np.fabs(rng.randn(int(num_poses)) * sigma_sc + 1.)  #scale factor
                offset = rng.randn(int(num_poses), 3) * sigma_com # translation factor
                rot_random = np.random.uniform(0, 360, 1) # rotation factor
                aug_model = list(random.sample(range(1, 6), 1))[0] # augmentation model factor

                depth_norm = list()
                joint_norm = list()
                if aug_model == 1:
                    # translation
                    depth_norm, joint_norm = translationHand(ori_depth, cube_size,
                                                             joint_uvd[id,34], offset[0],
                                                             joint_xyz[id, joint_id])
                elif aug_model == 2:
                    # rotation
                    depth_norm, joint_norm = rotateHand(ori_depth,
                                                        cube_size,
                                                        joint_uvd[id, 34], rot_random[0],
                                                        joint_xyz[id, joint_id], pad_value=0)
                elif aug_model == 3:
                    # scale
                    depth_norm, joint_norm = scaleHand(ori_depth, cube_size,
                                                       sc[0], joint_uvd[id,34],
                                                       joint_xyz[id, joint_id])
                elif aug_model == 4:
                    # scale, translation
                    depth_norm, joint_norm = translationHand(ori_depth, cube_size*sc[0],
                                                             joint_uvd[id, 34], offset[0],
                                                             joint_xyz[id, joint_id])
                elif aug_model == 5:
                    # scale, rotation
                    depth_norm, joint_norm = rotateHand(ori_depth,
                                                        cube_size*sc[0],
                                                        joint_uvd[id, 34], rot_random[0],
                                                        joint_xyz[id, joint_id], pad_value=0)
                elif aug_model == 6:
                    # scale, translation, rotation
                    com = joint3DToImg(jointImgTo3D(joint_uvd[id,34]) + offset[0])
                    depth_norm, joint_norm = rotateHand(ori_depth,
                                                        cube_size*sc[0],
                                                        com, rot_random[0],
                                                        joint_xyz[id, joint_id], pad_value=0)

                batch_image_train.append(depth_norm)
                batch_label_train.append(joint_norm)

            idx_batch = list(random.sample(range(0, batch_size), batch_size))
            batch_image = np.array(batch_image_train)[idx_batch].reshape(-1,128,128,1)
            batch_joint = np.array(batch_label_train).reshape(-1,42)[idx_batch]

            # fig = figure_joint_skeleton(batch_image[13].reshape(128,128), batch_joint[13].reshape(14,3), 1)
            # plt.show()
            # exit()

            loss = sess.run(loss_joints, feed_dict={X_in_image: batch_image, X_in_label: batch_joint,
                                                    keep_prob: 0.3})  # , Noise: batch_noise

            start_time = time.time()
            sess.run(optimizer, feed_dict={X_in_image: batch_image, X_in_label: batch_joint, keep_prob: 0.3})
            duration = time.time() - start_time

            if num_batch % 200 == 0:
                print("Epoch: %d, Step: %d,  Dis_loss: %f, Duration:%f sec"
                      % (epoch, num_batch, loss, duration))

            # save the loss of generator and discriminator
            steps.append(epoch * len(idx) // batch_size + num_batch)
            dis_loss_list.append(loss)

        saver.save(sess, "H:/HandPoseEstimation/model/nyu/model_{}.ckpt".format(epoch))
        average_error = test_model(dir, epoch)
        # if average_error <= average_error_flag:
        #     average_error_flag = average_error
        #     saver.save(sess, "H:/HandPoseEstimation/model/nyu/model_{}.ckpt".format(epoch))


duration_time_sum = time.time() - start_time_sum
print("The total training time: ",elapsed(duration_time_sum))


'''
#show the loss of generaor and discriminator at every batch
'''
fig = plt.figure(figsize=(8,6))
plt.plot(steps, dis_loss_list, label='dis_loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('The loss of train')
plt.legend()
plt.legend(loc = 'upper right')
plt.savefig(os.path.join(path, 'loss_curve.png'))











