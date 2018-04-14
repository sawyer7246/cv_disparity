import numpy as np
import cv2
import util as util
from matplotlib import pyplot as plt

path = 'data_pair/'
imgs = util.get_imgs_files(path, 'jpg')
# [(window_size,minDisparity,numDisparities,blockSize)]
params = [(10, 1, 16*3, 7), (10, 1, 16*3, 7), (7, 1, 16*3, 4), (7, 1, 16*2, 3), (3, 1, 16, 2)]

i = 0
ignore = []
for img_l, img_r, param in zip(imgs[::2], imgs[1::2], params):
    i += 1
    if i in ignore:
        continue
    imgL = cv2.imread(img_l, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(img_r, cv2.IMREAD_GRAYSCALE)

    imgL = cv2.resize(imgL, (600, 500), interpolation=cv2.INTER_CUBIC)
    imgR = cv2.resize(imgR, (600, 500), interpolation=cv2.INTER_CUBIC)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_title('left')
    ax1.imshow(imgL, cmap='gray')
    ax1 = fig.add_subplot(222)
    ax1.set_title('right')
    ax1.imshow(imgR, cmap='gray')

    # SGBM Parameters -----------------
    window_size = param[0]
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=param[1],
        numDisparities=param[2],  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=param[3],
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)

    ax1 = fig.add_subplot(223)
    ax1.set_title('disparity')
    ax1.imshow(filteredImg, cmap='gray')
    plt.show()
