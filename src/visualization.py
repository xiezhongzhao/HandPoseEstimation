import matplotlib.pyplot
import math
import numpy as np
import cv2
import os

def figure_joint_skeleton(dm, uvd_pt, flag):
    if flag:
        for i in range(len(uvd_pt)):
            uvd_pt[i, 0] = (uvd_pt[i, 0] + 1) / 2 * 128
            uvd_pt[i, 1] = (-uvd_pt[i, 1] + 1) / 2 * 128

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(dm, cmap=matplotlib.cm.gray) #matplotlib.cm.gray    matplotlib.cm.Greys
    ax.axis('off')

    fig_color = ['c', 'm', 'y', 'g', 'r']
    for f in range(5):
        ax.plot([uvd_pt[f*2,0], uvd_pt[f*2+1,0]],
                [uvd_pt[f*2,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=3)
        ax.scatter(uvd_pt[f*2,0],uvd_pt[f*2,1],s=60,c=fig_color[f])
        ax.scatter(uvd_pt[f*2+1,0],uvd_pt[f*2+1,1],s=60,c=fig_color[f])
        if f < 4:
            ax.plot([uvd_pt[13,0], uvd_pt[f*2+1,0]],
                    [uvd_pt[13,1], uvd_pt[f*2+1,1]], color=fig_color[f], linewidth=3)
    ax.plot([uvd_pt[9,0], uvd_pt[10,0]],
            [uvd_pt[9,1], uvd_pt[10,1]], color='r', linewidth=3)

    ax.scatter(uvd_pt[13,0], uvd_pt[13,1], s=200, c='w')
    ax.scatter(uvd_pt[11,0], uvd_pt[11,1], s=100, c='b')
    ax.scatter(uvd_pt[12,0], uvd_pt[12,1], s=100, c='b')

    ax.plot([uvd_pt[13,0], uvd_pt[11,0]],
            [uvd_pt[13,1], uvd_pt[11,1]], color='b', linewidth=3)
    ax.plot([uvd_pt[13,0], uvd_pt[12,0]],
            [uvd_pt[13,1], uvd_pt[12,1]], color='b', linewidth=3)
    ax.plot([uvd_pt[13,0], uvd_pt[10,0]],
            [uvd_pt[13,1], uvd_pt[10,1]], color='r', linewidth=3)

    return fig


def figure_smp_pts(dm, uvd_pt1, uvd_pt2):

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(dm, cmap=matplotlib.cm.Greys)
    ax.axis('off')

    for f in range(5):
        ax.plot([uvd_pt1[f * 2, 0], uvd_pt1[f * 2 + 1, 0]],
                [uvd_pt1[f * 2, 1], uvd_pt1[f * 2 + 1, 1]], color='r', linewidth=2)
        ax.scatter(uvd_pt1[f * 2, 0], uvd_pt1[f * 2, 1], s=50, c='r')
        ax.scatter(uvd_pt1[f * 2 + 1, 0], uvd_pt1[f * 2 + 1, 1], s=50, c='r')

        ax.plot([uvd_pt2[f * 2, 0], uvd_pt2[f * 2 + 1, 0]],
                [uvd_pt2[f * 2, 1], uvd_pt2[f * 2 + 1, 1]], color='b', linewidth=2)
        ax.scatter(uvd_pt2[f * 2, 0], uvd_pt2[f * 2, 1], s=50, c='b')
        ax.scatter(uvd_pt2[f * 2 + 1, 0], uvd_pt2[f * 2 + 1, 1], s=50, c='b')

        if f < 4:
            ax.plot([uvd_pt1[13, 0], uvd_pt1[f * 2 + 1, 0]],
                    [uvd_pt1[13, 1], uvd_pt1[f * 2 + 1, 1]], color='r', linewidth=2)
            ax.plot([uvd_pt2[13, 0], uvd_pt2[f * 2 + 1, 0]],
                    [uvd_pt2[13, 1], uvd_pt2[f * 2 + 1, 1]], color='b', linewidth=2)

    ax.plot([uvd_pt1[9, 0], uvd_pt1[10, 0]],
            [uvd_pt1[9, 1], uvd_pt1[10, 1]], color='r', linewidth=2)
    ax.plot([uvd_pt2[9, 0], uvd_pt2[10, 0]],
            [uvd_pt2[9, 1], uvd_pt2[10, 1]], color='b', linewidth=2)

    ax.scatter(uvd_pt1[13, 0], uvd_pt1[13, 1], s=100, c='w')
    ax.scatter(uvd_pt1[11, 0], uvd_pt1[11, 1], s=50, c='r')
    ax.scatter(uvd_pt1[12, 0], uvd_pt1[12, 1], s=50, c='r')
    ax.scatter(uvd_pt2[13, 0], uvd_pt2[13, 1], s=100, c='w')
    ax.scatter(uvd_pt2[11, 0], uvd_pt2[11, 1], s=50, c='b')
    ax.scatter(uvd_pt2[12, 0], uvd_pt2[12, 1], s=50, c='b')

    ax.plot([uvd_pt1[13, 0], uvd_pt1[11, 0]],
            [uvd_pt1[13, 1], uvd_pt1[11, 1]], color='r', linewidth=2)
    ax.plot([uvd_pt1[13, 0], uvd_pt1[12, 0]],
            [uvd_pt1[13, 1], uvd_pt1[12, 1]], color='r', linewidth=2)
    ax.plot([uvd_pt1[13, 0], uvd_pt1[10, 0]],
            [uvd_pt1[13, 1], uvd_pt1[10, 1]], color='r', linewidth=2)

    ax.plot([uvd_pt2[13, 0], uvd_pt2[11, 0]],
            [uvd_pt2[13, 1], uvd_pt2[11, 1]], color='b', linewidth=2)
    ax.plot([uvd_pt2[13, 0], uvd_pt2[12, 0]],
            [uvd_pt2[13, 1], uvd_pt2[12, 1]], color='b', linewidth=2)
    ax.plot([uvd_pt2[13, 0], uvd_pt2[10, 0]],
            [uvd_pt2[13, 1], uvd_pt2[10, 1]], color='b', linewidth=2)


    return fig

def save_figure_joint_skeleton(depth, uvd_pt, joint_uvd, id, flag):

    cube_size = 340

    fx = 588.03
    fy = 587.07

    joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])

    u, v, d = joint_uvd[id, 34]
    zstart = d - cube_size / 2.
    zend = d + cube_size / 2.

    xstart = int(math.floor((u * d / fx - cube_size / 2.) / d * fx))  # 103
    xend = int(math.floor((u * d / fx + cube_size / 2.) / d * fx))  # 333
    ystart = int(math.floor((v * d / fy - cube_size / 2.) / d * fy))  # 172
    yend = int(math.floor((v * d / fy + cube_size / 2.) / d * fy))  # 402

    msk1 = np.bitwise_and(depth < zstart, depth != 0)  # shape: (230,230)
    msk2 = np.bitwise_and(depth > zend, depth != 0)  # shape: (230,230)

    depth[msk1] = zstart
    depth[msk2] = zend

    fig = None
    result_path = None
    path = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/6.Multi-ResNet/results/'
    if flag == 0:
        fig = figure_joint_skeleton(depth, uvd_pt, 0)
        result_path = path + 'NYU_plot_joints/plotted_img_predict/'
    elif flag == 1:
        fig = figure_joint_skeleton(depth, joint_uvd[id, joint_id], 0)
        result_path = path + 'NYU_plot_joints/plotted_img_gt/'
    elif flag == 2:
        fig = figure_smp_pts(depth, joint_uvd[id, joint_id], uvd_pt)
        result_path = path + 'NYU_plot_joints/gt_predict/'
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(depth, cmap=matplotlib.cm.gray)
    ax.set_xlim(xstart, xend)
    ax.set_ylim(ystart, yend)

    # fig.canvas.draw()
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # data = np.flip(data, 0)


    # fig = matplotlib.pyplot.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(data, cmap=matplotlib.cm.gray)
    ax.axis('off')
    fig.savefig(os.path.join(result_path, '{}.png'.format(id)), bbox_inches='tight')

    # cv2.imwrite(os.path.join(result_path, '%id.jpg'%id), data)
