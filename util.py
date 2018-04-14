import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# util funtion
def fix_cv2_read_img_file(file):
    return fix_cv2_imread(cv2.imread(file))


def fix_cv2_imread(img_cv):
    img_rgb = np.zeros(img_cv.shape, img_cv.dtype)
    img_rgb[:, :, 0] = img_cv[:, :, 2]
    img_rgb[:, :, 1] = img_cv[:, :, 1]
    img_rgb[:, :, 2] = img_cv[:, :, 0]
    return img_rgb


def plot_img(img):
    plt.imshow(fix_cv2_imread(img))


def plot_k_imgs(files, k=10):
    for i in range(k):
        plt.title('Image {0},path:{1}'.format(i, files[i]))
        plt.imshow(fix_cv2_imread(cv2.imread(files[i], cv2.IMREAD_COLOR)))
        plt.show()


def get_imgs_files(path, file_type):
    """
    Get files
    :param path:
    :param file_type:
    :return:
    """
    imgs = []
    for root, dirs, tmp_files in os.walk(path):
        for i in tmp_files:
            if i.endswith(file_type):
                imgs.append(path + i)
    return imgs