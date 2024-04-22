import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def img_loader(data_folder: str, input_filename: str, output_filename: str):
    '''
    Read images from dataset folder, split them into X (input) and Y (output) data
    '''
    subfolders = sorted([f.path for f in os.scandir(data_folder) if f.is_dir()]) # List all subfolders in the data folder

    X, Y = [], []
    for s in subfolders:
        # Construct file paths for input and target images
        input_path = os.path.join(s, input_filename)
        output_path = os.path.join(s, output_filename)

        # Open and convert images to NumPy arrays
        input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        output_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

        # Append images to lists
        X.append(input_image)
        Y.append(output_image)

    # Convert lists to NumPy arrays
    X = np.array(X)
    Y = np.array(Y)

    print("X shape:", X.shape)
    print("y shape:", Y.shape)

    return X, Y

def phase_pair_img_loader(data_folder: str, input_filename_1: str, input_filename_2: str):
    '''
    Read images from dataset folder, X (input) contains a pair of images (vertical and horizontal phasemap)
    '''
    subfolders = sorted([f.path for f in os.scandir(data_folder) if f.is_dir()]) # List all subfolders in the data folder

    X = []
    for s in subfolders:
        # Construct file paths for input and target images
        input_path_1 = os.path.join(s, input_filename_1)
        input_path_2 = os.path.join(s, input_filename_2)

        # Open and convert images to NumPy arrays
        input_image_1 = cv2.imread(input_path_1, cv2.IMREAD_GRAYSCALE)
        input_image_2 = cv2.imread(input_path_2, cv2.IMREAD_GRAYSCALE)

        # Append images to lists
        X.append(np.dstack((input_image_1, input_image_2)))

    # Convert lists to NumPy arrays
    X = np.array(X)

    print("X shape:", X.shape)

    return X

def single_shot_data_numpy(data_folder: str, filename: str) -> None:
    '''
    Read all single shot datapoints (input, X) and save this set of data in numpy array (*.npy)
    @data_folder: base data folder location
    @filename: file name of the single-shot pattern
    '''

    subfolders = sorted([f.path for f in os.scandir(data_folder) if f.is_dir()]) # List all subfolders in the data folder

    single_shot_list = []
    for s in subfolders:
        single_shot_path = os.path.join(s, filename) # single-shot img path
        sinhgle_shot_img = cv2.imread(single_shot_path, cv2.IMREAD_GRAYSCALE) # read img
        single_shot_list.append(sinhgle_shot_img)

    np.save(os.path.join(data_folder, 'single_shot.npy'), np.stack(single_shot_list)) # save in npy

def numpy_loader(data_folder: str, numpy_filename: str):
    '''
    Read images that are saved in numpy array format
    '''

    X = np.load(os.path.join(data_folder, 'single_shot.npy'))
    Y = np.load(os.path.join(data_folder, numpy_filename))

    return X, Y

def img_read_crop(img_filepath: str, x: int, y: int, reshape: bool) -> np.array:
    '''
    Read single image for model prediction input, crop into 512 by 512 to match with tensor size
    x,y coordinate is the bound starting coordinate
    '''

    input_image = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE) # read img in gray
    input_image = input_image[y:y+512, x:x+512] # crop image
    # cv2.imwrite('input.png', input_image) # save cropped input image
    if reshape: 
        input_image = input_image.reshape(1, 512, 512, 1) # reshape to match with tensor size

    return input_image
