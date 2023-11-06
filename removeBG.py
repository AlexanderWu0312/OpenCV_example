from rembg import remove
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow

input_path = 'wu.jpg'
output_path = 'output.png'
input = Image.open(input_path)
output = remove(input)
output.save(output_path)
output.show()
img =cv2.imread('output.png')
cv2_imshow(img)
