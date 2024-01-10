'''
General single-period unwrapping to generate ground truth (GT) dataset
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def phase_map_calculate(img_paths: list) -> np.array:
    '''
    Naive phase map calculation: arctan [ (I2 - I4) / (I1 - I3) ]
    '''

    # read images in grayscale and convert uint8 into float value
    img_1 = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_2 = cv2.imread(img_paths[1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_3 = cv2.imread(img_paths[2], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_4 = cv2.imread(img_paths[3], cv2.IMREAD_GRAYSCALE).astype(np.float32)

    phase_map = np.arctan2((img_2 - img_4), (img_1 - img_3)) # add epsilon = 1e-10 to denominator if required (division by 0)
    phase_map += np.pi # add pi to the output because np.artan2 is from -pi to +pi, make it into 0 to 2pi

    return phase_map

def phase_map_calculate_zero_padding(img_paths: list) -> np.array:
    '''
    Zero-padding applied phase map calculation, for simulated data
    For denominator and numerator of arctan equation, if they have 0 (black) area within the pattern area, then convert the pixel value into 1 (0-255 range)
    '''

    # read images in grayscale and convert uint8 into float value
    img_1 = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_2 = cv2.imread(img_paths[1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_3 = cv2.imread(img_paths[2], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_4 = cv2.imread(img_paths[3], cv2.IMREAD_GRAYSCALE).astype(np.float32)

    a = (img_2 - img_4) # numerator
    white_pixels = a != 0
    a[white_pixels] = [255] # if not 0 (black, intended background zero-padded), make it into white (255) value

    contours, hierarchy = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the segmented image
    min_contour_area = 1  # Adjust this threshold based on your image and requirements
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area] # Filter out small contours (noise) based on a threshold area

    contour_mask = np.zeros_like(a)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, thickness=cv2.FILLED) # Create a mask for the filtered contours
    black_pixels_mask = (contour_mask != 0) # Create a binary mask for the pixels in the original image where the contour mask is black

    # Update the original image based on the masks
    numerator = np.copy(img_2 - img_4)
    numerator[black_pixels_mask & (numerator == 0)] = 1 # Set pixels to 1 where contour mask is black and original pixel value is 0

    b = (img_1 - img_3) # denominator
    white_pixels = b != 0
    b[white_pixels] = [255] # if not 0 (black, intended background zero-padded), make it into white (255) value

    contours, hierarchy = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the segmented image
    min_contour_area = 1  # Adjust this threshold based on your image and requirements
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area] # Filter out small contours (noise) based on a threshold area

    contour_mask = np.zeros_like(b)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, thickness=cv2.FILLED) # Create a mask for the filtered contours
    black_pixels_mask = (contour_mask != 0) # Create a binary mask for the pixels in the original image where the contour mask is black

    # Update the original image based on the masks
    denominator = np.copy(img_1 - img_3)
    denominator[black_pixels_mask & (denominator == 0)] = 1 # Set pixels to 1 where contour mask is black and original pixel value is 0

    phase_map = np.arctan2(numerator, denominator) # phase map calculation
    phase_map_normalized = np.where(phase_map == 0, 0, cv2.normalize(phase_map, None, 1, 255, cv2.NORM_MINMAX)) # convert phase map into 1-255 scale, keep the 0 as zero padding

    return phase_map_normalized

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

def minmax_normalize(img: np.ndarray) -> np.ndarray:
    '''
    Convert [0-2pi] scale into [0-255] scale
    @img: 0-2pi range image
    @return: 0-255 scaled image
    '''

    result = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return result

def save_unwrapped_phase(sub_folder: str, phase_map_vertical: np.array, phase_map_horizontal: np.array, vertical_file_name: str, horizontal_file_name: str):
    # save phase unwrapped files (0-2pi)
    cv2.imwrite(os.path.join(sub_folder, vertical_file_name), phase_map_vertical)
    cv2.imwrite(os.path.join(sub_folder, horizontal_file_name), phase_map_horizontal)

    # save minmax-normalized images (0-255)
    vertical_norm_file_name = vertical_file_name.split('.')[0] + '_norm.' + vertical_file_name.split('.')[1]
    vertical_minmax = cv2.normalize(phase_map_vertical, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(sub_folder, vertical_norm_file_name), vertical_minmax)
    horizontal_norm_file_name = horizontal_file_name.split('.')[0] + '_norm.' + horizontal_file_name.split('.')[1]
    horizontal_minmax = cv2.normalize(phase_map_horizontal, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(sub_folder, horizontal_norm_file_name), horizontal_minmax)

def save_unwrapped_phase_zero_padding(sub_folder: str, phase_map_vertical: np.array, phase_map_horizontal: np.array, vertical_file_name: str, horizontal_file_name: str):
    # save minmax-normalized images (0-255), normalization done in zero-padding logic
    vertical_norm_file_name = vertical_file_name.split('.')[0] + '_norm.' + vertical_file_name.split('.')[1]
    cv2.imwrite(os.path.join(sub_folder, vertical_norm_file_name), phase_map_vertical)
    horizontal_norm_file_name = horizontal_file_name.split('.')[0] + '_norm.' + horizontal_file_name.split('.')[1]
    cv2.imwrite(os.path.join(sub_folder, horizontal_norm_file_name), phase_map_horizontal)

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

    ''' Naive Phase Calculation Logic '''
    vertical_numpy_list, horizontal_numpy_list = [], []
    vertical_norm_numpy_list, horizontal_norm_numpy_list = [], []

    for s in tqdm(sub_folders):
        # get single period unwrapped phase maps
        phase_map_vertical = phase_map_calculate(read_vertical_4_phase(s))
        phase_map_horizontal = phase_map_calculate(read_horizontal_4_phase(s))

        # apply minmax normalization to scale [0-2pi] -> [0-255]
        phase_map_vertical_norm = minmax_normalize(phase_map_vertical)
        phase_map_horizontal_norm = minmax_normalize(phase_map_horizontal)

        # to compose a numpy list
        vertical_numpy_list.append(phase_map_vertical)
        horizontal_numpy_list.append(phase_map_horizontal)
        vertical_norm_numpy_list.append(phase_map_vertical_norm)
        horizontal_norm_numpy_list.append(phase_map_horizontal_norm)

        # save unwrapped phase in 8-bit image file
        save_unwrapped_phase(s, phase_map_vertical, phase_map_horizontal, 'img_9.png', 'img_10.png')

    # construct a numpy list, shape=(n, 512, 512) then save in numpy array
    np.save(os.path.join(base_path, 'vertical_raw.npy'), np.stack(vertical_numpy_list))
    np.save(os.path.join(base_path, 'horizontal_raw.npy'), np.stack(horizontal_numpy_list))
    np.save(os.path.join(base_path, 'vertical_norm.npy'), np.stack(vertical_norm_numpy_list))
    np.save(os.path.join(base_path, 'horizontal_norm.npy'), np.stack(horizontal_norm_numpy_list))
        
    ''' Zero-Padding Phase Map '''
    for s in tqdm(sub_folders):
        # get single period unwrapped phase maps
        phase_map_vertical = phase_map_calculate_zero_padding(read_vertical_4_phase(s))
        phase_map_horizontal = phase_map_calculate_zero_padding(read_horizontal_4_phase(s))

        # save unwrapped phase
        save_unwrapped_phase_zero_padding(s, phase_map_vertical, phase_map_horizontal, 'img_9.png', 'img_10.png')