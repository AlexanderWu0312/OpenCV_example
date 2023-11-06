import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# 讀取圖片
image = cv2.imread('wu.jpg')

# 檢查圖像是否成功讀取
if image is not None:
    # 模糊處理
    blurred = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # 灰度化
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 邊緣檢測
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # 卡通效果
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    # 保存處理後的圖片
    cv2.imwrite('cartoon.jpg', cartoon)

    # 顯示處理後的圖片
    cv2_imshow(cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("圖像無法讀取。")
