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

def draw_plot_save_phase_profile(sub_folder: str, phase_map_vertical: np.array, phase_map_horizontal: np.array, column: int, row: int):
    # color the column and row with max value (2pi)
    vertical_profile = phase_map_vertical[:, column].copy()
    phase_map_vertical[:, column] = 2 * np.pi
    horizontal_profile = phase_map_horizontal[row, :].copy()
    phase_map_horizontal[row, :] = 2 * np.pi

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(8,6))
    p0 = axs[0,0].imshow(phase_map_vertical, cmap='gray')
    fig.colorbar(p0, ax=axs[0,0])
    axs[0,0].set_title('Vertical Unwrapped Phase (Column=' + str(column) + ')')
    axs[0,1].plot(vertical_profile)

    p1 = axs[1,0].imshow(phase_map_horizontal, cmap='gray')
    fig.colorbar(p1, ax=axs[1,0])
    axs[1,0].set_title('Horizontal Unwrapped Phase (Row=' + str(row) + ')')
    axs[1,1].plot(horizontal_profile)

    fig.tight_layout() # to avoid axes overlapping
    fig.savefig(os.path.join(sub_folder, 'phase_map_profile.png')) # save

def draw_plot_save_singleshot_profile(sub_folder: str, img_file_name: str, column: int, row: int):
    # read single shot pattern img
    img_1 = cv2.imread(os.path.join(sub_folder, img_file_name), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_2 = cv2.imread(os.path.join(sub_folder, img_file_name), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # color the column and row with max value (white)
    vertical_profile = img_1[:, column].copy()
    img_1[:, column] = 255
    horizontal_profile = img_2[row, :].copy()
    img_2[row, :] = 255

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(8,6))
    p0 = axs[0,0].imshow(img_1, cmap='gray')
    fig.colorbar(p0, ax=axs[0,0])
    axs[0,0].set_title('Vertical Single-Shot (Column=' + str(column) + ')')
    axs[0,1].plot(vertical_profile)

    p1 = axs[1,0].imshow(img_2, cmap='gray')
    fig.colorbar(p1, ax=axs[1,0])
    axs[1,0].set_title('Horizontal Single-Shot (Row=' + str(row) + ')')
    axs[1,1].plot(horizontal_profile)

    fig.tight_layout() # to avoid axes overlapping
    fig.savefig(os.path.join(sub_folder, 'single_shot_profile.png')) # save

if __name__ == '__main__':
    base_path = './dl_data_set/dl_deflec_eye/'
    sub_folders = read_folder(base_path, 999)
    for s in tqdm(sub_folders):
        # get single period unwrapped phase maps
        phase_map_vertical = phase_map_calculate(read_vertical_4_phase(s))
        phase_map_horizontal = phase_map_calculate(read_horizontal_4_phase(s))

        # unwrapped phase map profile info save
        draw_plot_save_phase_profile(s, phase_map_vertical, phase_map_horizontal, 150, 150)

        # single shot pattern profile info save
        draw_plot_save_singleshot_profile(s, 'img_8.png', 150, 150)
        