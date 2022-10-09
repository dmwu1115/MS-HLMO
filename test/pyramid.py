from HS_HLMO.gaussian_pyramid import *

gaussian_pyramid = GaussianPyramid(2, 2, 1.6, 2)
image = cv2.imread('../images/image3.png', cv2.IMREAD_GRAYSCALE)
pyramid = gaussian_pyramid.generate_pyramid(image)

for o, octave in enumerate(pyramid):
    for l, layer in enumerate(octave):
        win = "octaver_" + str(o) + "_layer_" + str(l)
        cv2.imshow(win, layer)
cv2.waitKey(0)
