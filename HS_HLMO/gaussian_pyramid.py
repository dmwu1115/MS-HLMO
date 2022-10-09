import cv2
from scipy import ndimage

class GaussianPyramid():
    def __init__(self, nOctaves, nLayers, sigma, downsample_resize):
        self.nOctaves = nOctaves
        self.nLayers = nLayers
        self.sigma = sigma
        self.resize = 1 / downsample_resize

    def generate_pyramid(self, image) -> list:
        """
        生成高斯金字塔
        :param image: 单通道图像
        :return: pyramid[octaves][layers]
        """
        pyramid = []

        layer = []
        for l in range(self.nLayers):
            if l == 0:
                layer.append(image)
            else:
                layer.append(ndimage.gaussian_filter(layer[l - 1], sigma=self.sigma))

        for o in range(self.nOctaves):
            if o == 0:
                pyramid.append(layer)
            else:
                octave = []
                for l in range(self.nLayers):
                    octave.append(cv2.resize(pyramid[o - 1][l], (-1, -1), fx=self.resize, fy=self.resize,
                                             interpolation=cv2.INTER_LINEAR))
                pyramid.append(octave)
        return pyramid