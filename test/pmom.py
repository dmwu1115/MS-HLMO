import math
import cv2
from HS_HLMO.preprocess import *
from HS_HLMO.pmom_generator import *
from HS_HLMO.ggloh_describer import *
import numpy as np

def nothing(val):
    pass

processer = PreProcesser()

image = cv2.imread('../images/image3.png', cv2.IMREAD_GRAYSCALE)
image = processer.process(image)

cv2.namedWindow('pmom', cv2.WINDOW_NORMAL)
cv2.createTrackbar('pmom_sigma', 'pmom', 0, 100, nothing)

while True:
    sigma = cv2.getTrackbarPos('pmom_sigma', 'pmom') / 10
    pmom = PMOM(sigma)
    ori = pmom.generate_PMOM(image)
    ori = ori / math.pi * 255.0
    ori = ori.astype(np.uint8)

    cv2.imshow('pmom', ori)
    if cv2.waitKey(1) == ord('q'):
        break
