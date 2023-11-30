import os
import numpy as np
import cv2

def img_loader(data_folder: str, input_filename: str, output_filename: str):
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