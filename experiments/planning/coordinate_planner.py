import cv2
import numpy as np
from collections import deque
from math import sqrt

lower_black = np.array([0, 0, 0])  # HSV下限
upper_black = np.array([150, 255, 100])  # HSV上限

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])


# Ramer - Douglas - Peucker算法
def rdp(points, epsilon):
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        results = rdp(points[:index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[end]]
    return results


# 计算点到直线的垂直距离
def perpendicular_distance(point, start, end):
    if start == end:
        return sqrt((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2)
    else:
        n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d


list = []


def get_list(data):
    return data


def red(hsv):  # 红色掩膜
    lower_red_1 = np.array((0, 120, 70))
    upper_red_1 = np.array((10, 255, 255))
    lower_red_2 = np.array((170, 120, 70))
    upper_red_2 = np.array((180, 255, 255))
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    return mask_red


def blue(hsv):  # 蓝色掩膜
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # 去噪
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.erode(mask_blue, kernel, iterations=2)  # 腐蚀去除小噪点
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=5)  # 膨胀填充空洞
    return mask_blue


def road(hsv, lower_color, upper_color):  # 路的掩膜
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # 去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)  # 腐蚀去除小噪点
    mask = cv2.dilate(mask, kernel, iterations=10)  # 膨胀填充空洞
    return mask


def draw_rect(image, contours):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # 过滤小面积噪点
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def detect_color_object(ori_image, ori_hsv):
    # 转换为HSV颜色空间
    mask_black_road = road(ori_hsv, lower_black, upper_black)
    mask_red_start = red(ori_hsv)
    mask_blue_end = blue(ori_hsv)
    # 转换成单一颜色
    color_mask = np.zeros_like(ori_image)
    color_mask[mask_red_start > 0] = (1, 1, 250)
    color_mask[mask_black_road > 0] = (0, 100, 0)
    color_mask[mask_blue_end > 0] = (254, 1, 1)
    # 保存结果
    cv2.imwrite("processed_image.jpg", color_mask)


def find_to_start(hsv):  # 寻找起点
    red_mask = red(hsv)
    red_points = np.column_stack(np.where(red_mask > 0))
    if len(red_points) > 0:
        start = tuple(red_points[0][::-1])
    else:
        raise ValueError("未找到起始点（红色）")
    return start


def find_to_end(hsv):  # 寻找终点
    blue_mask = blue(hsv)
    blue_points = np.column_stack(np.where(blue_mask > 0))
    if len(blue_points) > 0:
        end = tuple(blue_points[0][::-1])
    else:
        raise ValueError("未找到终点（蓝色）")
    return end


def find_obstacle(hsv):  # 识别障碍物（黑色）
    lower_black = np.array([35, 43, 46])
    upper_black = np.array([77, 255, 255])
    obstacle_mask = cv2.inRange(hsv, lower_black, upper_black)
    return obstacle_mask


def find_way(start, end, obstacle_mask):
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

    return found, parent, current


def point_of_way(found, parent):
    if found:
        path = []
        current = end
        while current != start:
            path.append(current)
            current = parent[current]
        path.append(start)
        path.reverse()

        # 使用RDP算法平滑路径
        epsilon = 10  # 可以调整这个值来控制平滑程度
        smoothed_path = rdp(path, epsilon)
        print(smoothed_path)
        # ----显示路径
        # 创建路径矩阵
        path_matrix = np.zeros((height, width), dtype=np.uint8)  # 创建全零数组
        for point in smoothed_path:
            path_matrix[point[1], point[0]] = 255  # 点
        # 在路径矩阵上绘制折线
        smoothed_path = np.array(smoothed_path, np.int32)
        smoothed_path = smoothed_path.reshape((-1, 1, 2))
        # 显示原始图像和路径
        for point in smoothed_path:
            point = tuple(point[0])  # 将 point 转换为长度为 2 的元组
            cv2.circle(ori_image, point, 3, (0, 255, 255), -1)  # 黄色路径
        # 将路径点连接起来
        cv2.polylines(ori_image, [smoothed_path], False, (0, 255, 0), 2)
        # 缩放图像以适应屏幕
        scale_factor = 0.7  # 缩小50%，可根据需要调整
        new_dim = (int(ori_image.shape[1] * scale_factor), int(ori_image.shape[0] * scale_factor))
        resized_image = cv2.resize(ori_image, new_dim, interpolation=cv2.INTER_AREA)

        # 显示缩放后的图像
        cv2.imshow('Rasterized Image with Path', resized_image)
        # ----------------------------------------------------
        return smoothed_path


def save_path_to_file(path_points, filename='path_points.txt'):
    try:
        if not path_points:
            print("警告：路径点列表为空，无需保存。")
            return

        for idx, point in enumerate(path_points):
            # 处理 numpy 数组（转为一维并校验长度）
            if isinstance(point, np.ndarray):
                point = point.flatten()  # 展平为一维数组
                if point.size < 2:
                    raise IndexError(f"第 {idx + 1} 个点坐标维度不足")
            else:
                if len(point) < 2:
                    raise IndexError(f"第 {idx + 1} 个点格式错误")

            # 提取坐标（兼容数组和列表）
            x = point[0]
            y = point[1]
            with open(filename, 'a') as f:  # 建议用 'a' 避免重复覆盖
                f.write(f"{x}, {y}\n")  # 逐点写入，避免批量处理歧义

        print(f"路径已成功保存到 {filename}")

    except IndexError as e:
        print(f"索引错误：{e}，请检查坐标点格式是否为 (x,y)")
    except Exception as e:
        print(f"保存失败：{e}")

if __name__ == '__main__':
    ori_image = cv2.imread('road 5.jpg')
    ori_hsv = cv2.cvtColor(ori_image, cv2.COLOR_BGR2HSV)
    detect_color_object(ori_image, ori_hsv)  # 查找处理图像并储存为processed_image

    processed_image = cv2.imread('processed_image.jpg')
    hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

    height, width, _ = processed_image.shape

    start = find_to_start(hsv)
    print(start)  # 正常
    end = find_to_end(hsv)
    print(end)  # 正常
    obstacle_mask = find_obstacle(hsv)

    found, parent, current = find_way(start, end, obstacle_mask)
    print(found)  # 正常

    points = point_of_way(found, parent)

    save_path_to_file(points, filename='path_points.txt')

    cv2.waitKey(0)
    cv2.destroyAllWindows()