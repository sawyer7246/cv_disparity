import numpy as np
import cv2
from matplotlib import pyplot as plt
import util as util

path = 'data_pair/'
imgs = util.get_imgs_files(path, 'jpg')
# util.plot_k_imgs(imgs)

rect_list = [(11, 15, 480, 460), (430, 450, 967, 760), (280, 340, 1330, 830), (461, 164, 950, 930), (216, 93, 890, 460)]

for image, rect in zip(imgs[::2], rect_list):
    img = util.fix_cv2_read_img_file(image)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # rect = (2, 25, 500, 430)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    cv2.rectangle(img, rect[:2], rect[2:], (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()
