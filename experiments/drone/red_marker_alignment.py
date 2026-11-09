import cv2
import numpy as np
import time
import os
from djitellopy import Tello, TelloException

# 颜色范围（HSV）红色物体
RED_LOWER1 = np.array([0, 100, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

# 绿色准心参数
CROSSHAIR_SIZE = 15  # 准心线条长度（像素）
CROSSHAIR_THICKNESS = 2  # 准心线条粗细
CROSSHAIR_COLOR = (0, 255, 0)


def force_land(drone) -> bool:
    """强制降落函数，增加重试机制，返回是否成功降落"""
    print("尝试强制降落...")
    for i in range(5):  # 最多尝试5次降落
        try:
            drone.land()
            time.sleep(3)  # 等待降落动作完成（无人机需要时间执行降落）
            print(f"降落尝试 {i + 1}/5 成功")
            return True
        except TelloException as te:
            print(f"降落尝试 {i + 1}/5 失败（Tello协议异常）: {te}，2秒后重试...")
            time.sleep(2)
        except Exception as e:
            print(f"降落尝试 {i + 1}/5 失败（其他异常）: {e}，2秒后重试...")
            time.sleep(2)
    print("警告：所有降落尝试失败，请立即手动干预！")
    return False


def detect_red_square(rgb_frame):
    #转换图像到HSV颜色空间
    #hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)

    #创建红色的掩码
    red_mask1 = cv2.inRange(rgb_frame, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(rgb_frame, RED_LOWER2, RED_UPPER2)
    combined_mask = cv2.bitwise_or(red_mask1, red_mask2)

    #形态学操作（去除噪声，填充小空洞，确保轮廓完整）
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # 闭运算：先膨胀再腐蚀，填充空洞
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)  # 开运算：先腐蚀再膨胀，去除小噪声

    #寻找轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #筛选红色正方形
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < 200:  # 面积阈值：可根据红色正方形实际大小调整
            continue

        # 轮廓近似（将不规则轮廓简化为近似多边形）
        contour_perimeter = cv2.arcLength(contour, True)  # 计算轮廓周长（闭合轮廓）
        approx_polygon = cv2.approxPolyDP(contour, 0.04 * contour_perimeter, True)  # 近似多边形

        # 正方形判断条件：4条边 + 宽高比接近1（0.9~1.1）
        if len(approx_polygon) == 4:
            x, y, w, h = cv2.boundingRect(approx_polygon)  # 获取轮廓的轴对齐边界框
            aspect_ratio = float(w) / h  # 计算宽高比
            if 0.9 <= aspect_ratio <= 1.1:  # 宽高比接近1，判定为正方形
                center_x = x + w // 2  # 正方形中心x坐标
                center_y = y + h // 2  # 正方形中心y坐标
                return (x, y, w, h), (center_x, center_y)

    # 未检测到符合条件的红色正方形
    return None, None


def draw_green_crosshair(rgb_frame):
    """
    在视频流正中心绘制绿色准心（十字形）
    返回：绘制准心后的帧、准心中心坐标(center_x, center_y)
    """
    frame_height, frame_width = rgb_frame.shape[:2]  # 获取帧的高度和宽度
    crosshair_center_x = frame_width // 2  # 准心中心x（帧宽度的一半）
    crosshair_center_y = frame_height // 2  # 准心中心y（帧高度的一半）

    # 绘制准心的水平线条（左右方向）
    cv2.line(
        rgb_frame,
        (crosshair_center_x - CROSSHAIR_SIZE, crosshair_center_y),  # 左端点
        (crosshair_center_x + CROSSHAIR_SIZE, crosshair_center_y),  # 右端点
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS
    )

    # 绘制准心的垂直线条（上下方向）
    cv2.line(
        rgb_frame,
        (crosshair_center_x, crosshair_center_y - CROSSHAIR_SIZE),  # 上端点
        (crosshair_center_x, crosshair_center_y + CROSSHAIR_SIZE),  # 下端点
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS
    )

    return rgb_frame, (crosshair_center_x, crosshair_center_y)


def adjust_drone_position(drone, red_center, crosshair_center, frame_width):

    # 计算水平方向偏移量（dx：红色中心x - 准心中心x）
    dx = red_center[0] - crosshair_center[0]

    # 对齐阈值：偏移小于此像素值，认为已对齐（可根据精度需求调整）
    ALIGN_THRESHOLD = 8  # 建议范围：5~10像素
    # 移动速度：避免速度过快导致过冲（Tello速度范围：1~100，建议30~50）
    MOVE_SPEED = 40

    # 标记是否执行了移动操作
    has_moved = False

    # 水平方向调整（左右移动）
    if dx > ALIGN_THRESHOLD:
        # 红色正方形在准心右侧 → 无人机向右移动
        print(f"向右移动（偏移量：{dx}像素）")
        drone.send_rc_control(MOVE_SPEED, 0, 0, 0)  # RC控制：左右(+右/-左)，前后，上下，旋转
        has_moved = True
    elif dx < -ALIGN_THRESHOLD:
        # 红色正方形在准心左侧 → 无人机向左移动
        print(f"向左移动（偏移量：{dx}像素）")
        drone.send_rc_control(-MOVE_SPEED, 0, 0, 0)
        has_moved = True

    # 前后方向调整（基于红色正方形大小：确保正方形在画面中尺寸稳定，避免过近/过远）
    # 计算红色正方形宽度占帧宽度的比例（比例范围：0.1~0.3为宜，避免过大/过小）
    red_square_width = red_center[2]  # 从边界框中获取正方形宽度（原检测函数返回的边界框第3个参数是w）
    width_ratio = red_square_width / frame_width

    if width_ratio < 0.15:
        # 正方形过小（距离过远）→ 无人机向前移动
        print(f"向前移动（正方形占比：{width_ratio:.2f}，目标占比：0.15~0.25）")
        drone.send_rc_control(0, MOVE_SPEED // 2, 0, 0)  # 前向速度减半，避免过快
        has_moved = True
    elif width_ratio > 0.25:
        # 正方形过大（距离过近）→ 无人机向后移动
        print(f"向后移动（正方形占比：{width_ratio:.2f}，目标占比：0.15~0.25）")
        drone.send_rc_control(0, -MOVE_SPEED, 0, 0)
        has_moved = True

    # 如果执行了移动，等待移动完成后悬停
    if has_moved:
        time.sleep(0.6)  # 等待移动（根据速度调整，速度越快，时间越短）
        drone.send_rc_control(0, 0, 0, 0)  # 停止移动，进入悬停
        time.sleep(0.4)  # 悬停稳定时间

    # 判断是否已对齐（水平偏移小于阈值 + 正方形尺寸在目标范围）
    is_aligned = (abs(dx) <= ALIGN_THRESHOLD) and (0.15 <= width_ratio <= 0.25)
    return is_aligned


def main():
    # 确保dji文件夹存在
    if not os.path.exists("dji"):
        os.makedirs("dji")
        print("已创建dji文件夹用于保存照片")
    else:
        print("使用已存在的dji文件夹保存照片")

    #初始化无人机
    drone = Tello()
    emergency_landed = False  # 标记是否执行过紧急降落
    photo_counter = 0  # 照片计数器（用于命名）
#2
    try:
        #连接无人机
        print("\n正在连接Tello无人机...")
        drone.connect()
        battery = drone.get_battery()
        print(f"无人机电池电量：{battery}%")

        # 电池低电量保护（低于15%不执行任务，避免空中断电）
        if battery < 15:
            print("警告：电池电量低于15%，终止任务！")
            return

        #开启视频流并初始化
        print("\n开启无人机摄像头...")
        drone.streamon()
        time.sleep(4)  # 等待视频流初始化
        frame_reader = drone.get_frame_read()

        # 检查视频流是否有效（增加重试机制）
        is_frame_valid = False
        rgb_frame = None  # 存储RGB格式的有效帧
        for retry in range(3):
            bgr_frame = frame_reader.frame
            if bgr_frame is not None:
                #BGR转RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                is_frame_valid = True
                break
            print(f"视频流获取失败（重试 {retry + 1}/3），2秒后重新初始化...")
            drone.streamoff()
            time.sleep(2)
            drone.streamon()
            time.sleep(3)
            frame_reader = drone.get_frame_read()

        if not is_frame_valid:
            print("错误：视频流初始化失败，无法获取摄像头画面，终止任务！")
            force_land(drone)
            return

        #创建视频显示窗口
        frame_height, frame_width = rgb_frame.shape[:2]
        cv2.namedWindow("Tello Drone View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tello Drone View", 960, 720)  # 调整窗口大小（便于观察）
        print("\n已开启实时画面窗口（按 'q' 可手动退出）")

        #无人机起飞并悬停
        print("\n无人机准备起飞...")
        drone.takeoff()
        time.sleep(3)  # 等待起飞并上升到安全高度（约1.2米）
        drone.send_rc_control(0, 0, 0, 0)  # 确保起飞后悬停
        time.sleep(1)
        print("无人机已起飞并进入悬停状态")

        #寻找红色正方形并调整位置至中心对齐
        print("\n开始寻找红色正方形...")
        is_centered = False  # 标记是否已中心对齐
        max_attempts = 60  # 最大调整尝试次数（避免无限循环，每次尝试约1秒）
        attempt_count = 0  # 当前尝试次数

        while not is_centered and attempt_count < max_attempts:
            attempt_count += 1
            # 获取当前帧
            current_frame = frame_reader.frame
            if current_frame is None:
                print(f"调整尝试 {attempt_count}/{max_attempts}：未获取到有效帧，跳过...")
                time.sleep(1)
                continue

            # 绘制绿色准心
            frame_with_crosshair, crosshair_center = draw_green_crosshair(current_frame.copy())

            # 检测红色正方形
            red_bbox, red_center = detect_red_square(current_frame)

            # 处理检测结果
            if red_bbox and red_center:
                # 绘制红色正方形边界框和中心标记
                x, y, w, h = red_bbox
                cv2.rectangle(frame_with_crosshair, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色边框
                cv2.circle(frame_with_crosshair, (red_center[0], red_center[1]), 5, (255, 0, 0), -1)  # 蓝色中心点
                cv2.putText(
                    frame_with_crosshair,
                    "Red Square",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                # 调整无人机位置（将红色中心对准准心）
                # 补充边界框宽度到red_center中（用于前后调整）
                red_center_with_width = (red_center[0], red_center[1], w)
                is_centered = adjust_drone_position(drone, red_center_with_width, crosshair_center, frame_width)

                if is_centered:
                    print(f"调整尝试 {attempt_count}/{max_attempts}：红色正方形已与准心中心重合！")
            else:
                # 未检测到红色正方形：缓慢旋转寻找（顺时针旋转）
                print(f"调整尝试 {attempt_count}/{max_attempts}：未检测到红色正方形，顺时针旋转寻找...")
                drone.send_rc_control(0, 0, 0, 30)  # RC控制：旋转(+顺时针/-逆时针)
                time.sleep(1)
                drone.send_rc_control(0, 0, 0, 0)  # 停止旋转
                time.sleep(0.5)

            # 显示实时画面（按 'q' 手动退出）
            cv2.imshow("Tello Drone View", frame_with_crosshair)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户手动按下 'q'，终止任务并执行紧急降落...")
                force_land(drone)
                return

        # 检查是否成功对齐（超过最大尝试次数则紧急降落）
        if not is_centered:
            print(f"\n错误：超过最大尝试次数（{max_attempts}次），未能将红色正方形与准心对齐，执行紧急降落！")
            force_land(drone)
            return

        # 对齐后悬停并拍摄第一张照片
        print("\n在对齐位置悬停2秒...")
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(2)

        print("拍摄第一张照片...")
        first_photo_frame = frame_reader.frame
        if first_photo_frame is not None:
            photo_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")  # 时间戳（避免照片重名）
            first_photo_path = os.path.join("dji", f"red_square_{timestamp}_{photo_counter}.jpg")
            # 转换为RGB格式保存（OpenCV默认读取为BGR，需转RGB才能保证颜色正常）
            first_photo_rgb = cv2.cvtColor(first_photo_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(first_photo_path, first_photo_rgb)
            print(f"第一张照片已保存：{first_photo_path}")
        else:
            print("警告：无法获取有效帧，第一张照片拍摄失败！")
        time.sleep(2)  # 拍摄后等待2秒，确保画面稳定

        # 9. 向前飞行100cm并悬停
        print("\n执行向前飞行100厘米...")
        drone.move_forward(100)  # Tello的move_forward单位为厘米
        time.sleep(3)  # 等待飞行完成（根据距离和速度，100cm约需3秒）
        drone.send_rc_control(0, 0, 0, 0)  # 飞行后悬停稳定
        time.sleep(2)
        print("已向前飞行100厘米，进入悬停状态")

        # 10. 拍摄第二张照片
        print("拍摄第二张照片...")
        second_photo_frame = frame_reader.frame
        if second_photo_frame is not None:
            photo_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            second_photo_path = os.path.join("dji", f"rgb_photo_{timestamp}_{photo_counter}.jpg")
            second_photo_rgb = cv2.cvtColor(second_photo_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(second_photo_path, second_photo_rgb)
            print(f"第二张照片已保存：{second_photo_path}")
        else:
            print("警告：无法获取有效帧，第二张照片拍摄失败！")
        time.sleep(2)

        # 11. 任务完成，执行正常降落
        print("\n所有任务完成，准备正常降落...")
        if not emergency_landed:
            emergency_landed = force_land(drone)

    # 异常处理：捕获无人机协议异常和其他未知异常
    except TelloException as te:
        print(f"\n无人机协议异常：{str(te)}，执行紧急降落...")
        if not emergency_landed:
            emergency_landed = force_land(drone)
    except Exception as e:
        print(f"\n未知错误：{str(e)}，执行紧急降落...")
        if not emergency_landed:
            emergency_landed = force_land(drone)
    # 最终清理：无论是否异常，都关闭视频流、窗口和无人机连接
    finally:
        print("\n开始清理资源...")
        # 关闭无人机视频流
        try:
            drone.streamoff()
            print("无人机视频流已关闭")
        except Exception as e:
            print(f"关闭视频流时出错：{str(e)}")

        # 关闭OpenCV显示窗口
        cv2.destroyAllWindows()
        print("OpenCV显示窗口已关闭")

        # 终止无人机连接
        try:
            drone.end()
            print("无人机连接已终止")
        except Exception as e:
            print(f"终止无人机连接时出错：{str(e)}")

        # 打印最终状态
        if emergency_landed:
            print("提示：已执行紧急降落逻辑")
        print("程序结束，照片已保存至dji文件夹")

if __name__ == "__main__":
    main()