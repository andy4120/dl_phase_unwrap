'''
General single-period unwrapping to generate ground truth (GT) dataset
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def phase_map_calculate(img_paths: list) -> np.array:
    # read images in grayscale and convert uint8 into float value
    img_1 = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_2 = cv2.imread(img_paths[1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_3 = cv2.imread(img_paths[2], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_4 = cv2.imread(img_paths[3], cv2.IMREAD_GRAYSCALE).astype(np.float32)

    phase_map = np.arctan2((img_2 - img_4), (img_1 - img_3)) # add epsilon = 1e-10 to denominator if required (division by 0)
    phase_map += np.pi # add pi to the output because np.artan2 is from -pi to +pi, make it into 0 to 2pi

    return phase_map

def read_folder(base_path: str, max_data: int) -> list:
    '''
    With a given base path (sub-folders contained), reutrn a subfolder list that contains 4-phase shifted imgs in both x, y directions
    '''
    result = []
    for i in range(max_data + 1):
        char_length = len(str(i))
        prefix = '0' * (5 - char_length)
        result.append(os.path.join(base_path, prefix + str(i)))
    return result

def read_vertical_4_phase(base_path: str) -> list:
    # join with base folder path (e.g. './dl_data_set/dl_deflec_eye/00000/')
    imgpath_1 = os.path.join(base_path, 'img_0.png')
    imgpath_2 = os.path.join(base_path, 'img_1.png')
    imgpath_3 = os.path.join(base_path, 'img_2.png')
    imgpath_4 = os.path.join(base_path, 'img_3.png')

    return [imgpath_1, imgpath_2, imgpath_3, imgpath_4]

def read_horizontal_4_phase(base_path: str) -> list:
    # join with base folder path (e.g. './dl_data_set/dl_deflec_eye/00000/')
    imgpath_1 = os.path.join(base_path, 'img_4.png')
    imgpath_2 = os.path.join(base_path, 'img_5.png')
    imgpath_3 = os.path.join(base_path, 'img_6.png')
    imgpath_4 = os.path.join(base_path, 'img_7.png')

    return [imgpath_1, imgpath_2, imgpath_3, imgpath_4]

if __name__ == '__main__':
    base_path = './dl_data_set/dl_deflec_eye/'
    sub_folders = read_folder(base_path, 999)
    for s in tqdm(sub_folders):
        phase_map_vertical = phase_map_calculate(read_vertical_4_phase(s))
        phase_map_horizontal = phase_map_calculate(read_horizontal_4_phase(s))

        plt.imshow(phase_map_vertical, cmap='gray')
        plt.show()
        plt.imshow(phase_map_horizontal, cmap='gray')
        plt.show()