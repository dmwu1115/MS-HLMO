import cv2

img = cv2.imread('./image2.png')
img = cv2.resize(img,(-1, -1), fx=0.5, fy=0.5)
cv2.imwrite('./image2_0.5scale.png', img)