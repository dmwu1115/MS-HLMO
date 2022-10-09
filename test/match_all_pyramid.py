import time

from HS_HLMO.KeypointDescriber import *
from HS_HLMO.KeypointDetector import *
from HS_HLMO.gaussian_pyramid import *
from HS_HLMO.preprocess import *
from HS_HLMO.matching import *

image1 = cv2.imread('../images/image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('../images/image2.png', cv2.IMREAD_GRAYSCALE)

preprocess = PreProcesser()
pyramid = GaussianPyramid(2, 3, 1.6, 2)
detector = KeypointsDetector(1e-7)
describer = KeypointDescriber(1.6, 12, 12, 5)
matcher = FeatureMatching()

image1 = preprocess.process(image1, 1.6)
image2 = preprocess.process(image2, 1.6)

kpts1 = detector.detect(image1)
kpts2 = detector.detect(image2)

img_pyramid1 = pyramid.generate_pyramid(image1)
img_pyramid2 = pyramid.generate_pyramid(image2)

desc1s, desc2s = [], []
t = time.time()
for octave1, octave2 in zip(img_pyramid1, img_pyramid2):
    for layer1, layer2 in zip(octave1, octave2):
        desc1s.append(describer.generate_descriptors(layer1, kpts1))
        desc2s.append(describer.generate_descriptors(layer2, kpts2))
print(f'DescribeCost: {time.time() - t}')
print(f'AvgDescribeCost: {(time.time() - t) / (len(img_pyramid1) * len(img_pyramid1[0]))}')

t = time.time()
allmatches = matcher.match(desc1s, desc2s)
print(f'MatchCost: {time.time() - t}')
print()
