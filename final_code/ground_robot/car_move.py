import cv2
import numpy as np
import time
import threading
import queue
from robomaster import robot
from robomaster.camera import EPCamera
from robomaster import blaster
from robomaster.gimbal import Gimbal
from robomaster.chassis import Chassis
from ultralytics import YOLO

# 机器人初始化
host = "192.168.2.1"
port = 40923
ep_robot = robot.Robot()
ep_robot.initialize()
ep_camera = EPCamera(ep_robot)
ep_blaster = ep_robot.blaster
ep_gimbal = Gimbal(ep_robot)
ep_chassis = Chassis(ep_robot)

frame_queue = queue.Queue(maxsize=1)  # 线程安全的帧队列
model = YOLO('yolov8n.pt')  # 加载YOLOv8模型


# 在屏幕中心绘制准心
def draw_cross(img):
    h, w = img.shape[:2]
    color = (0, 255, 0)
    cv2.line(img, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), color, 2)
    cv2.line(img, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), color, 2)
    cv2.circle(img, (w // 2, h // 2), 10, color, 2)
    return img, w // 2, h // 2


# 摄像头线程：持续捕获帧并放入队列
def camera_thread_func():
    ep_camera.start_video_stream(display=False)
    try:
        while True:
            frame = ep_camera.read_cv2_image(timeout=1, strategy='pipeline')
            if frame is not None:
                if frame_queue.full():
                    frame_queue.get_nowait()  # 丢弃旧帧
                frame_queue.put(frame.copy())  # 放入新帧
    except Exception as e:
        print(f"摄像头线程异常: {e}")
    finally:
        ep_camera.stop_video_stream()


def load_path_from_file(filename='path_points.txt'):
    """从文本文件加载路径点并返回元组列表"""
    path_points = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                # 跳过注释行
                if line.startswith('#'):
                    continue

                # 解析坐标点
                line = line.strip()
                if not line:
                    continue

                # 分割x和y坐标
                parts = line.split(',')
                if len(parts) != 2:
                    print(f"警告: 无法解析行 '{line}'")
                    continue

                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    # 存储为整数元组
                    path_points.append((int(x), int(y)))
                except ValueError as e:
                    print(f"解析坐标时出错: {e}")

    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    if not path_points:
        print("警告: 文件中未找到有效坐标点")
        return None

    print(f"成功加载路径，包含 {len(path_points)} 个点")

    return path_points


def load_points_from_file(filename):
    """从文本文件加载路径点并返回元组列表"""
    points = []
    with open(filename, 'r') as f:
        for line in f:
            nums=line.strip().split()
            points.extend(float(num) for num in nums)
    return points

def move_on_path():
    
    distance=load_points_from_file(filename='path_distance.txt')
    angles=load_points_from_file(filename='angles.txt')
    '''flag=0
    if angles[0]<0:
        flag=1'''
    
    for a,d in zip(angles,distance):
        '''if flag==1:
            t=a
        else:
            t=-a'''
        ep_chassis.move(x=0,y=0,z=-a,xy_speed=0,z_speed=40).wait_for_completed()#z速度范围10-540°/s   转弯 
        
        ep_chassis.move(x=d,y=0,z=0,xy_speed=1,z_speed=0).wait_for_completed()#xy速度范围0.5-2m/s  直走
            

# 控制机器人云台跟踪目标
def track_object(x, y, x_center, y_center):
    # 水平方向调整
    if x < x_center - 30:
        ep_gimbal.move(pitch=0, yaw=-3, pitch_speed=30, yaw_speed=30).wait_for_completed()
    elif x > x_center + 30:
        ep_gimbal.move(pitch=0, yaw=3, pitch_speed=30, yaw_speed=30).wait_for_completed()

    # 垂直方向调整
    if y < y_center - 20:
        ep_gimbal.move(pitch=2, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
    elif y > y_center + 20:
        ep_gimbal.move(pitch=-2, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()


def main():
    
    ep_chassis.move(x=0.65,y=0,z=0,xy_speed=1,z_speed=0).wait_for_completed()
    ep_chassis.move(x=0,y=0,z=90,xy_speed=0,z_speed=40).wait_for_completed()
    
    move_on_path()
    # 启动摄像头线程
    threading.Thread(target=camera_thread_func, daemon=True).start()

    try:
        fired = False  # 标记是否已经发射过
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame, x_center, y_center = draw_cross(frame)  # 绘制准心

                # 使用YOLOv8检测物体
                results = model(frame)
                if results:  # 检查results是否为空
                    annotated_frame = results[0].plot()
                    person_detected = False
                    person_x, person_y = None, None

                    for box in results[0].boxes:
                        if int(box.cls) == 0 and box.conf > 0.5:  # 0是person的类别ID
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            person_x = (x1 + x2) // 2
                            person_y = (y1 + y2) // 2
                            person_detected = True

                            # 在目标周围绘制额外标记
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            confidence = box.conf.item()  # 将Tensor转换为浮点数
                            cv2.putText(annotated_frame, f"Person {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            break

                    # 显示处理后的帧
                    cv2.imshow("YOLOv8 Person Detection", annotated_frame)
                    person_found=0
                    # 如果检测到人物，进行跟踪
                    if person_detected:
                        person_found=1
                        track_object(person_x, person_y, x_center, y_center)

                        # 检查目标是否在准心附近且未发射过
                        if (abs(person_x - x_center) < 40 and abs(person_y - y_center) < 30) and not fired:
                            ep_blaster.fire(fire_type=blaster.WATER_FIRE)
                            fired = True
                            print("发射!")
                    if person_found == 0:
                        ep_gimbal.move(pitch=0, yaw=10, pitch_speed=0, yaw_speed=40).wait_for_completed()
                else:
                    cv2.imshow("YOLOv8 Person Detection", frame)
            # 按ESC退出
            key = cv2.waitKey(1) & 0xFF
            if key == "q":
                break
    finally:
        cv2.destroyAllWindows()
        ep_robot.close()


if __name__ == '__main__':
    main()