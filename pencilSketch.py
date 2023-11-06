import numpy as np
import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread( "wu.jpg", -1 )
img1, img2 = cv2.pencilSketch( img )
cv2_imshow( img )
cv2_imshow(img1 )
cv2_imshow(img2 )
cv2.waitKey( 0 )
