import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def calculation_metrics(img1: np.ndarray, img2: np.ndarray) -> None:
    '''
    Fcn call MSE, PSNR quantitative metrics, prints out values
    '''
    print('MSE:', mse(img1, img2))
    print('PSNR:', psnr(img1, img2))

def pred_profile_plot(pred_img: np.ndarray, direction: str, index: int, plot_save_path: str) -> None:
    '''
    Prediction only (no GT, usually real data) profile plot
    '''

    if direction == 'col' or direction == 'vertical':
        # column = 250 # real
        # column = 25 # test
        gt_img[:, index] = 2 * np.pi
        pred_profile = pred_img[:, index].copy()
    elif direction == 'row' or direction == 'horizontal':
        # row = 350 # real
        # row = 25 # test
        gt_img[index, :] = 2 * np.pi
        pred_profile = pred_img[index, :].copy()

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    p0 = axs[0].imshow(pred_img, cmap='gray')
    fig.colorbar(p0, ax=axs[0])

    if direction == 'col' or direction == 'vertical':
        axs[0].set_title('Vertical Phase Profile (Column=' + str(index) + ')')
    elif direction == 'row' or direction == 'horizontal':
        axs[0].set_title('Horizontal Phase Profile (Row=' + str(index) + ')')

    axs[1].plot(pred_profile, label='Predicted', alpha=0.7)
    axs[1].set_title('Predicted Profile')
    axs[1].set_xlabel('Pixel')
    axs[1].set_ylabel('Intensity')
    axs[1].legend()

    fig.tight_layout() # to avoid axes overlapping
    fig.savefig(plot_save_path) # save

def gt_pred_profile_plot(gt_img: np.ndarray, pred_img: np.ndarray, direction: str, index: int, plot_save_path: str) -> None:
    '''
    Test data GT vs. Pred profile comparison plot
    @direction: either 'row' or 'col' (depends on horizontal/vertical phase)
    @index: row/col index value
    '''

    if direction == 'col' or direction == 'vertical':
        # column = 250 # real
        # column = 25 # test
        gt_profile = gt_img[:, index].copy()
        gt_img[:, index] = 2 * np.pi
        pred_profile = pred_img[:, index].copy()
    elif direction == 'row' or direction == 'horizontal':
        # row = 350 # real
        # row = 25 # test
        gt_profile = gt_img[index, :].copy()
        gt_img[index, :] = 2 * np.pi
        pred_profile = pred_img[index, :].copy()

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    p0 = axs[0].imshow(gt_img, cmap='gray')
    fig.colorbar(p0, ax=axs[0])

    if direction == 'col' or direction == 'vertical':
        axs[0].set_title('Vertical Phase Profile (Column=' + str(index) + ')')
    elif direction == 'row' or direction == 'horizontal':
        axs[0].set_title('Horizontal Phase Profile (Row=' + str(index) + ')')

    axs[1].plot(gt_profile, label='Ground Truth', alpha=0.7)
    axs[1].plot(pred_profile, label='Predicted', alpha=0.7)
    axs[1].set_title('Ground Truth vs. Predicted Profiles')
    axs[1].set_xlabel('Pixel')
    axs[1].set_ylabel('Intensity')
    axs[1].legend()

    fig.tight_layout() # to avoid axes overlapping
    fig.savefig(plot_save_path) # save

if __name__ == '__main__':
    gt = './img_9_norm.png'
    pred = './horizontal_norm.png'
    gt_img = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)

    calculation_metrics(gt_img, pred_img)
    gt_pred_profile_plot(gt_img, pred_img, direction='horizontal', index=25, plot_save_path='./model/result.png') # test data profile plot
    pred_profile_plot(pred_img, direction='horizontal', index=25, plot_save_path='./model/result.png') # real data prediction only profile plot