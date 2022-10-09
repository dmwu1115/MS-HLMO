import cv2
import numpy as np
from scipy import ndimage


class PreProcesser():
    @staticmethod
    def process(image: np.ndarray, sigma=1.6) -> np.ndarray:
        """
        预处理图像，去噪+Normalize
        :param image: BGR or Gray
        :return: 像素值在(0,1)之间的float的单通道图
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        image = ndimage.gaussian_filter(image, sigma)
        return image
