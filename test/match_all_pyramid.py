import time

from HS_HLMO.KeypointDescriber import *
from HS_HLMO.KeypointDetector import *
from HS_HLMO.gaussian_pyramid import *
from HS_HLMO.preprocess import *
from HS_HLMO.matching import *

octave_num1, octave_num2 = 2, 4
layer_num1, layer_num2 = 2, 4

image1 = cv2.imread('../images/image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('../images/image2_0.5scale.png', cv2.IMREAD_GRAYSCALE)

preprocess = PreProcesser()
pyramid = GaussianPyramid(octave_num1, layer_num1, 1.6, 2)
pyramid2 = GaussianPyramid(octave_num2, layer_num2, 1.6, 2)
detector = KeypointsDetector(1e-7)
describer = KeypointDescriber(1.6, 12, 12, 5)
matcher = FeatureMatching()

image1 = preprocess.process(image1, 1.6)
image2 = preprocess.process(image2, 1.6)

kpts1 = detector.detect(image1)
kpts2 = detector.detect(image2)

img_pyramid1 = pyramid.generate_pyramid(image1)
img_pyramid2 = pyramid2.generate_pyramid(image2)

desc1s, desc2s = [], []
t = time.time()
for octave1 in img_pyramid1:
    for layer1 in octave1:
        desc1s.append(describer.generate_descriptors(layer1, kpts1))
for octave2 in img_pyramid2:
    for layer2 in octave2:
        desc2s.append(describer.generate_descriptors(layer2, kpts2))
print(f'DescribeCost: {time.time() - t}')
print(f'AvgDescribeCost: {(time.time() - t) / (len(img_pyramid1) * len(img_pyramid1[0]))}')

t = time.time()
allmatches = matcher.match(desc1s, desc2s)
res = matcher.remove_outlier_matches(allmatches, octave_num1, layer_num1, octave_num2, layer_num2)
print(f'MatchCost: {time.time() - t}')

# Draw Matches
kpts1 = HS_HLMO.norm_coord.denormalize_coord(kpts1, image1.shape[1], image1.shape[0])
kpts2 = HS_HLMO.norm_coord.denormalize_coord(kpts2, image2.shape[1], image2.shape[0])

keypoints1 = []
keypoints2 = []
for kpt in kpts1:
    keypoints1.append(cv2.KeyPoint(kpt[0], kpt[1], 7, 0))
for kpt in kpts2:
    keypoints2.append(cv2.KeyPoint(kpt[0], kpt[1], 7, 0))

image1 *= 255.0
image2 *= 255.0
image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR).astype(np.uint8)
image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR).astype(np.uint8)

matchimg = cv2.drawMatches(image1, keypoints1, image2, keypoints2, res, None, matchColor=(0, 255, 0),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('match', matchimg)
cv2.waitKey(0)