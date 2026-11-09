import cv2
import numpy as np
import time
from djitellopy import Tello
import threading

# ============== 初始化配置 ==============
drone = Tello()
drone.connect()
drone.streamon()
drone.set_speed(30)
time.sleep(2)

# S型搜索参数
s_search_step = 50  # 每次横向移动距离(cm)
s_search_length = 100  # S型总长度(cm)
s_search_direction = 1  # 初始搜索方向(1=右, -1=左)
s_search_progress = 0  # 当前搜索进度(cm)

# 显示控制标志
display_active = True


# ============== 视觉识别模块 ==============
def find_red_square(frame):
    """检测红色正方形并返回中心坐标和面积"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array((0, 120, 70))
    upper_red1 = np.array((10, 255, 255))
    lower_red2 = np.array((170, 120, 70))
    upper_red2 = np.array((180, 255, 255))

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.6 <= aspect_ratio <= 1.4:
                center_x = x + w // 2
                center_y = y + h // 2
                target = (center_x, center_y, w * h)
                break
    return target


# ============== 位置调整模块 ==============
def adjust_position(target_center, frame_width):
    """根据目标中心调整无人机位置"""
    cx, cy, _ = target_center
    error_x = cx - frame_width // 2

    if abs(error_x) > 20:
        speed = int(error_x * 0.3)
        if speed > 0:
            drone.move_right(abs(speed))
        else:
            drone.move_left(abs(speed))
        return False
    return True


# ============== 图片保存模块 ==============
def save_photo(frame, filename):
    """保存图片到指定路径"""
    # 修正：添加文件名和路径
    path = f"drone_photos/{filename}.jpg"
    cv2.imwrite(path, frame)
    print(f"图片已保存至: {path}")


# ============== S型搜索函数 ==============
def s_search():
    """执行S型搜索模式"""
    global s_search_direction, s_search_progress

    # 横向移动
    drone.move_right(s_search_step * s_search_direction)
    time.sleep(1.5)

    # 更新搜索进度
    s_search_progress += s_search_step

    # 检查是否需要转向
    if s_search_progress >= s_search_length:
        # 重置进度，转向
        s_search_progress = 0
        s_search_direction *= -1
        # 纵向移动（创建S型）
        drone.move_forward(s_search_step)
        time.sleep(1.5)


# ============== 显示线程函数 ==============
def display_thread():
    """独立线程用于显示无人机摄像头画面"""
    global display_active

    cv2.namedWindow("Drone Live View", cv2.WINDOW_NORMAL)

    try:
        while display_active:
            if drone.stream_on:
                frame = drone.get_frame_read().frame

                # 添加状态信息
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - frame_read.time)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Drone Live View", frame)

            # 按 'q' 键退出显示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"显示线程错误: {str(e)}")

    finally:
        cv2.destroyWindow("Drone Live View")


# ============== 主飞行逻辑 ==============
try:
    # 创建并启动显示线程
    display_t = threading.Thread(target=display_thread)
    display_t.daemon = True
    display_t.start()

    drone.takeoff()
    drone.move_up(80)  # 升高到合适高度

    print("开始寻找红色正方形...")
    frame_read = drone.get_frame_read()
    frame_width = frame_read.frame.shape[1]
    aligned = False
    search_attempts = 0
    max_search_attempts = 20  # 最大搜索次数

    while not aligned and search_attempts < max_search_attempts:
        frame = frame_read.frame
        target = find_red_square(frame)

        if target:
            print("检测到红色正方形，正在对齐...")
            aligned = adjust_position(target, frame_width)
            # 在主窗口标记目标
            cv2.circle(frame, (target[0], target[1]), 10, (0, 255, 0), -1)
        else:
            print("未检测到目标，执行S型搜索...")
            s_search()
            search_attempts += 1

        # 显示主窗口（与显示线程中的窗口不同）
        cv2.imshow("Drone View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if aligned:
        save_photo(frame_read.frame, "photo1_aligned")

        # 后续移动和拍照逻辑
        movements = [
            ("向右", drone.move_right, 20),
            ("向后", drone.move_back, 20),
            ("向左", drone.move_left, 20)
        ]

        for idx, (direction, func, distance) in enumerate(movements, 2):
            print(f"第{idx}步：{direction}移动{distance}cm")
            func(distance)
            time.sleep(1.5)
            save_photo(frame_read.frame, f"photo{idx}_{direction}")
    else:
        print("搜索超时，未找到目标")

except Exception as e:
    print(f"飞行异常：{str(e)}")
    drone.emergency()

finally:
    # 停止显示线程
    display_active = False
    time.sleep(0.5)  # 等待线程结束

    cv2.destroyAllWindows()
    drone.streamoff()
    drone.land()
    drone.end()
    print("任务完成，无人机已降落")