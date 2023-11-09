'''
General Unwrap Method
1. skimage.restoration.unwrap_phase
2. multi-frequency based unwrap
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def phase_map_calculate(img_1_path: str, img_2_path: str, img_3_path: str, img_4_path: str):
    # read images in grayscale and convert uint8 into float value
    img_1 = cv2.imread(img_1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_2 = cv2.imread(img_2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_3 = cv2.imread(img_3_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_4 = cv2.imread(img_4_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    phase_map = np.arctan2((img_2 - img_4), (img_1 - img_3)) # add epsilon = 1e-10 to denominator if required (division by 0)
    phase_map += np.pi # add pi to the output because np.artan2 is from -pi to +pi, make it into 0 to 2pi

    return phase_map

if __name__ == '__main__':
    img_base_path = './dl_data_set/dl_deflec_eye/00000'

    # imgpath_1 = os.path.join(img_base_path, 'img_0.png')
    # imgpath_2 = os.path.join(img_base_path, 'img_1.png')
    # imgpath_3 = os.path.join(img_base_path, 'img_2.png')
    # imgpath_4 = os.path.join(img_base_path, 'img_3.png')

    imgpath_1 = os.path.join(img_base_path, 'img_4.png')
    imgpath_2 = os.path.join(img_base_path, 'img_5.png')
    imgpath_3 = os.path.join(img_base_path, 'img_6.png')
    imgpath_4 = os.path.join(img_base_path, 'img_7.png')

    phase_map = phase_map_calculate(imgpath_1, imgpath_2, imgpath_3, imgpath_4)

    plt.imshow(phase_map, cmap='gray')
    plt.show()