import time

from HS_HLMO.KeypointDetector import *
from HS_HLMO.KeypointDescriber import *
from HS_HLMO.preprocess import *

image1 = cv2.imread('../images/image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('../images/image2.png', cv2.IMREAD_GRAYSCALE)

processer = PreProcesser()

detector = KeypointsDetector(1e-7, 10)
# detector = HarrisKeypointDetector()

describer = KeypointDescriber(1.6, 12, 12, R0=5)
matcher = cv2.BFMatcher()

image1 = processer.process(image1)
image2 = processer.process(image2)
kpts1 = detector.detect(image1)
kpts2 = detector.detect(image2)

t = time.time()
descriptors1 = describer.generate_descriptors(image1, kpts1)
print(f"DescribeCost1:{time.time()-t}")
t = time.time()
descriptors2 = describer.generate_descriptors(image2, kpts2)
print(f"DescribeCost2:{time.time()-t}")
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

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

def nothing(val):
    pass
cv2.namedWindow('match', cv2.WINDOW_NORMAL)
cv2.createTrackbar('ratio', 'match', 0, 100, nothing)

while True:
    ratio = cv2.getTrackbarPos('ratio', 'match') / 100.
    size = int(ratio * len(matches))
    good_matches = []
    for i in range(size):
        good_matches.append(matches[i])

    matchimg = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,matchColor=(0,255,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('match', matchimg)
    if cv2.waitKey(1) == ord('q'):
        break
print()