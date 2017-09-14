from depth_sharpener import sharpen_depth
import cv2

sz = 500

depth = cv2.imread('./depth.png')
depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
depth = depth.astype(float)

img = cv2.imread('./img.png')

d = sharpen_depth(img, depth)

d = (d - d.min()) / (d.max() - d.min())


while True:
    cv2.imshow('a', img)
    cv2.imshow('e', d)
    cv2.waitKey(1)
