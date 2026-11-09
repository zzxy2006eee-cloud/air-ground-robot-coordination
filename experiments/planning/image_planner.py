import cv2
import numpy as np
from collections import deque

lower_black = np.array([0, 0, 0])  # HSV下限
upper_black = np.array([150, 255, 100])  # HSV上限

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])


def mask(hsv, lower_color, upper_color):
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # 去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)  # 腐蚀去除小噪点
    mask = cv2.dilate(mask, kernel, iterations=10)  # 膨胀填充空洞
    return mask


def red(hsv):
    lower_red_1 = np.array((0, 120, 70))
    upper_red_1 = np.array((10, 255, 255))
    lower_red_2 = np.array((170, 120, 70))
    upper_red_2 = np.array((180, 255, 255))
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    return mask_red


def draw_rect(image, contours):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # 过滤小面积噪点
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def detect_color_object(image_path):
    image = cv2.imread(image_path)
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_black = mask(hsv, lower_black, upper_black)
    mask_blue = mask(hsv, lower_blue, upper_blue)
    mask_red = red(hsv)

    color_mask = np.zeros_like(image)
    color_mask[mask_red > 0] = (1, 1, 250)
    color_mask[mask_black > 0] = (0, 100, 0)
    color_mask[mask_blue > 0] = (254, 1, 1)
    # 查找轮廓
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制边界框
    draw_rect(image, contours_blue)
    draw_rect(image, contours_black)
    draw_rect(image, contours_red)

    # 显示结果
    cv2.imshow("Original", image)
    cv2.imshow("Mask", color_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 保存结果
    cv2.imwrite("result.jpg", color_mask)


def path(image_path):
    ori_image = cv2.imread('road mini.jpg')
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    # 转换为HSV颜色空间，更适合颜色识别
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 识别起始点（红色），红色在HSV中有两个范围
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([5, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_points = np.column_stack(np.where(red_mask > 0))
    if len(red_points) > 0:
        start = tuple(red_points[0][::-1])
    else:
        raise ValueError("未找到起始点（红色）")

    # 识别终点（蓝色）
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_points = np.column_stack(np.where(blue_mask > 0))
    if len(blue_points) > 0:
        end = tuple(blue_points[0][::-1])
    else:
        raise ValueError("未找到终点（蓝色）")

    # 识别障碍物（黑色）
    lower_black = np.array([35, 43, 46])
    upper_black = np.array([77, 255, 255])
    obstacle_mask = cv2.inRange(hsv, lower_black, upper_black)

    # 定义方向数组
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 广度优先搜索
    queue = deque([(start, 0)])  # 队列中存储 (当前位置, 路径长度)
    visited = set([start])
    parent = {}
    found = False

    while queue:
        current, distance = queue.popleft()
        if current == end:
            found = True
            break
        for dx, dy in directions:
            new_x, new_y = current[0] + dx, current[1] + dy
            if 0 <= new_x < width and 0 <= new_y < height \
                    and obstacle_mask[new_y, new_x] == 0 and (new_x, new_y) not in visited:
                queue.append(((new_x, new_y), distance + 1))
                visited.add((new_x, new_y))
                parent[(new_x, new_y)] = current

    # 绘制最短路径
    if found:
        path = []
        current = end
        while current != start:
            path.append(current)
            current = parent[current]
        path.append(start)
        path.reverse()
        for point in path:
            cv2.circle(ori_image, point, 3, (0, 255, 255), -1)  # 黄色路径

    # 显示结果
    cv2.imshow('Rasterized Image with Path', ori_image)


def main():
    detect_color_object('road mini.jpg')
    path('result.jpg')


if __name__ == "__main__":
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()