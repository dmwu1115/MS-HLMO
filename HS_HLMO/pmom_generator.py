import cv2
import numpy as np
from scipy import ndimage

class PMOM():
    def __init__(self, sigma):
        self.sigma = sigma

    def generate_PMOM(self, image):
        G_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        G_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        G_w_x = ndimage.gaussian_filter(G_x**2 - G_y**2, self.sigma)
        G_w_y = ndimage.gaussian_filter(2 * G_x * G_y, self.sigma)
        orientationMap = np.arctan2(G_w_y, G_w_x) / 2
        return orientationMap