import imutils
from skimage.measure import compare_ssim
import argparse
import cv2


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
# 	help="first input image")
# ap.add_argument("-s", "--second", required=True,
# 	help="second")
# args = vars(ap.parse_args())


# load the two input images
grayA = cv2.resize(cv2.imread('/Users/thangna/Downloads/comm/401/20171002_105822.jpg',0),(612,816))
grayB = cv2.resize(cv2.imread('/Users/thangna/Downloads/comm/401/20171002_105818.jpg',0),(612,816))

cv2.imshow('image A', grayA)
cv2.imshow('image B', grayB)
cv2.waitKey(0)
cv2.destroyAllWindows()

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


