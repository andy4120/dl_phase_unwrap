import os
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

def img_loader(data_folder: str, input_filename: str, output_filename: str) -> tuple(np.ndarray, np.ndarray):
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

def img_read_crop(img_filepath: str, x: int, y: int) -> np.ndarray:
    '''
    Read single image for model prediction input, crop into 512 by 512 to match with tensor size
    x,y coordinate is the bound starting coordinate
    '''

    input_image = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE) # read img in gray
    input_image = input_image[y:y+512, x:x+512] # crop image
    # cv2.imwrite('input.png', input_image) # save cropped input image
    input_for_prediction = input_image.reshape(1, 512, 512, 1) # reshape to match with tensor size

    return input_for_prediction

def mse(img1: np.ndarray, img2: np.ndarray):
    '''
    Read 2 images (GT, Pred) -> returns MSE value
    '''
    mse = np.mean(np.subtract(img1, img2) ** 2)
    return mse

def psnr(img1: np.ndarray, img2: np.ndarray):
    '''
    Read 2 images (GT, Pred) -> returns PSNR value
    '''
    # reference: https://dsp.stackexchange.com/a/50704
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculation_metrics(img1, img2) -> None:
    '''
    Fcn call MSE, PSNR quantitative metrics, prints out values
    '''
    print('MSE:', mse(img1, img2))
    print('PSNR:', psnr(img1, img2))
