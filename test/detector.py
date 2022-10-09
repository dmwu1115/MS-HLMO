from HS_HLMO.KeypointDetector import *
from HS_HLMO.preprocess import *
from HS_HLMO.norm_coord import *

detector = KeypointsDetector(1e-5, 7)
image = cv2.imread('../images/image1.png', cv2.IMREAD_GRAYSCALE)

res = detector.detect(image)
res = denormalize_coord(res, image.shape[1], image.shape[0])

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for pt in res:
    image = cv2.circle(image, pt, 2, (255, 255, 0), -1)

cv2.imshow('Harris', image)
cv2.waitKey(0)
print()