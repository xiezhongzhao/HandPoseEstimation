import numpy as np

dir = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/6.Multi-ResNet/'
def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, len(positions[0]) / 3, 3))


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu', 'msra'])


def get_dataset_file(dataset):
    return dir+'groundtruth/{}/{}_test_groundtruth_label.txt'.format(dataset, dataset)


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.036865, -587.075073, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


# def world2pixel(x, fx, fy, ux, uy):
#     x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
#     x[:, :, 1] = fy - fx*x[:, :, 1] / x[:, :, 2]
#     x[:, :, 2] = x[:, :, 2]
#     return x

def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = - x[:, :, 1] * fy / x[:, :, 2] + uy
    return x

def get_errors(dataset, in_file):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    labels = get_positions(get_dataset_file(dataset))
    outputs = get_positions(in_file)

    # if axis == 0:
    #    outputs = outputs + 6
    #
    # elif axis == 1:
    #    outputs = outputs + 4
    #
    # elif axis == 2:
    # outputs = outputs + 4
    #
    # elif axis == 3:
    #    outputs = outputs + 1

    params = get_param(dataset)
    labels = pixel2world(labels, *params)
    outputs = pixel2world(outputs, *params)
    #
    # print("labels.shape", labels[:, : ,0].reshape(-1,14,1).shape)
    # print("outputs.shape", outputs[:, :, :].shape)
    # exit()

    # sum = np.sum((labels[:,:,0:0] - outputs[:,:,0:0]) ** 2, axis=1)
    # print("np.sum", sum.shape)
    # exit()

    # errors_X = np.sqrt(np.sum((labels[:, :, axis].reshape(-1, 21, 1) - outputs[:, :, axis].reshape(-1, 21, 1)) ** 2, axis=2))
    # errors_Y = np.sqrt(np.sum((labels[:, :, 1].reshape(-1, 21, 1) - outputs[:, :, 1].reshape(-1, 21, 1)) ** 2, axis=2))
    # errors_Z = np.sqrt(np.sum((labels[:, :, 2].reshape(-1, 21, 1) - outputs[:, :, 2].reshape(-1, 21, 1)) ** 2, axis=2))

    # errors_mean = (errors_X + errors_Y + errors_Z) / 3.0

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    # print("errors: ",errors.shape)

    # errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    return errors

def get_msra_viewpoint(in_file):
    with open(in_file) as f:
        viewpoint = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(viewpoint), (-1, 2))










