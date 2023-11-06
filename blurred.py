import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# 讀取圖片
image = cv2.imread('wu.jpg')

# 模糊處理
blurred = cv2.GaussianBlur(image, (9, 9), 0)

# 調整亮度和對比度
alpha = 1.5  # 亮度增強因子
beta = 30    # 對比度增強因子
enhanced = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

# 顯示原始圖片
cv2_imshow(image)

# 顯示處理後的圖片
cv2_imshow(enhanced)

# 等待用戶按任意鍵並關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
