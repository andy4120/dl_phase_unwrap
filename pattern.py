import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_single_shot_pattern(plot=False):
    pattern = np.zeros((1080, 1920, 1))
    A = np.tile(0.5 + 0.25 * np.sin(np.linspace(0, 1 * (20 * np.pi), 1920)), (1080, 1))
    X = 0.5 + 0.25 * np.sin(np.linspace(0, 1 * (10 * np.pi), 1080))
    C = np.tile(X[:1080], (1920, 1)).T
    pattern[:, :, 0] = A + C

    pattern_255 = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('./single_shot.png', pattern_255)
    return pattern, pattern_255

if __name__ == '__main__':
    pattern, pattern_255 = generate_single_shot_pattern()