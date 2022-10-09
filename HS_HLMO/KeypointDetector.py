from HS_HLMO.norm_coord import normalize_coord
import cv2
import numpy as np
from scipy import ndimage

class KeypointsDetector():
    def __init__(self, thresh=1e-9, lnms_window_size=7):
        self.lnms_window_szie = lnms_window_size
        self.thresh = thresh

    def detect(self, image) -> list:
        """
        检测Harris角点
        :param image: 单通道图像
        :return: 归一化后的keypoints坐标，[0, 1]
        """
        harris_image = cv2.cornerHarris(image, 2, 3, k=0.04, borderType=cv2.BORDER_REFLECT)
        res_image = (self.__lnms(harris_image) == harris_image)
        kpts = []
        for y in range(res_image.shape[0]):
            for x in range(res_image.shape[1]):
                if res_image[y, x] and harris_image[y, x] > self.thresh:
                    kpts.append((x, y))

        kpts = normalize_coord(kpts, image.shape[1], image.shape[0])
        return kpts

    def __lnms(self, harris_image) -> np.ndarray:
        return ndimage.maximum_filter(harris_image, self.lnms_window_szie)


# use det(M)/tr(M) as the cornerness
class HarrisKeypointDetector():
    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        srcImage = srcImage[:] * 255.0
        sobel_x = ndimage.sobel(srcImage, 1, mode='reflect')
        sobel_y = ndimage.sobel(srcImage, 0, mode='reflect')
        Ix2 = ndimage.gaussian_filter(sobel_x ** 2, sigma=0.5)
        Iy2 = ndimage.gaussian_filter(sobel_y ** 2, sigma=0.5)
        IxIy = ndimage.gaussian_filter(sobel_x * sobel_y, sigma=0.5)

        harrisImage = (Ix2 * Iy2 - IxIy ** 2) / ((Ix2 + Iy2) + 1e-20)
        orientationImage = np.arctan2(sobel_y, sobel_x)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        maxImg = ndimage.maximum_filter(harrisImage, size=(7, 7))
        destImage = (maxImg == harrisImage)
        return destImage

    def detect(self, image):
        height, width = image.shape
        features = []

        harrisImage, orientationImage = self.computeHarrisValues(image)

        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x] or harrisImage[y, x] < 50:
                    continue

                f = (x / width, y / height)
                features.append(f)

        return features