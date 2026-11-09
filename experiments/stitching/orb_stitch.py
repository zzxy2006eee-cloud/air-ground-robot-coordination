import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('gongzhupai 1.jpg')
img2 = cv2.imread('gongzhupai 2.jpg')

# 初始化 ORB 检测器
orb = cv2.ORB_create()

# 检测关键点和计算描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 暴力匹配描述符
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 计算单应性矩阵（RANSAC 抗干扰）
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 透视变换拼接图像
result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

# 融合重叠区域（加权平均）
mask = np.zeros_like(result)
mask[0:img2.shape[0], 0:img2.shape[1]] = 255
result = cv2.seamlessClone(img2, result, mask, (img2.shape[1]//2, img2.shape[0]//2), cv2.NORMAL_CLONE)

# 设置缩小后的图像宽度和高度
new_width = int(result.shape[1] * 0.3)  # 这里设置为原宽度的比例
new_height = int(result.shape[0] * 0.3)  # 这里设置为原高度的比例

# 缩小图像
resized_result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 显示结果
cv2.imshow('Resized Result', resized_result)
cv2.waitKey(0)
cv2.destroyAllWindows()