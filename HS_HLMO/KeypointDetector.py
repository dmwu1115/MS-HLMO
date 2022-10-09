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
        height, width = image.shape

        harris_image = cv2.cornerHarris(image, 2, 3, k=0.04, borderType=cv2.BORDER_REFLECT)
        res_image = (self.__lnms(harris_image) == harris_image)
        mask = res_image & (harris_image > self.thresh)

        f = np.argwhere(mask).astype(np.float32)
        f_t = np.zeros_like(f)
        f_t[:, 0] = f[:, 1] / width
        f_t[:, 1] = f[:, 0] / height
        f_t = f_t.tolist()

        kpts = [tuple(x) for x in f_t]
        return kpts

    def __lnms(self, harris_image) -> np.ndarray:
        return ndimage.maximum_filter(harris_image, self.lnms_window_szie)


# use det(M)/tr(M) as the cornerness
class HarrisKeypointDetector():
    def __init__(self, thresh=50, lnms_window_size=7):
        self.lnms_window_szie = lnms_window_size
        self.thresh = thresh

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
        maxImg = ndimage.maximum_filter(harrisImage, size=self.lnms_window_szie)
        destImage = (maxImg == harrisImage)
        return destImage

    def detect(self, image):
        height, width = image.shape

        harrisImage, orientationImage = self.computeHarrisValues(image)

        harrisMaxImage = self.computeLocalMaxima(harrisImage)
        mask = harrisMaxImage & (harrisImage > self.thresh)

        f = np.argwhere(mask).astype(np.float32)
        f_t = np.zeros_like(f)
        f_t[:, 0] = f[:, 1] / width
        f_t[:, 1] = f[:, 0] / height
        f_t = f_t.tolist()

        kpts = [tuple(x) for x in f_t]
        return kpts