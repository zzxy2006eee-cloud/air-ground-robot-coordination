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
                    banana_detected = False
                    banana_x, banana_y = None, None

                    for box in results[0].boxes:
                        if int(box.cls) == 0 and box.conf > 0.5:  # 0是banana的类别ID
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            banana_x = (x1 + x2) // 2
                            banana_y = (y1 + y2) // 2
                            banana_detected = True

                            # 在目标周围绘制额外标记
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            confidence = box.conf.item()  # 将Tensor转换为浮点数
                            cv2.putText(annotated_frame, f"Banana {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            break

                    # 显示处理后的帧
                    cv2.imshow("YOLOv8 Banana Detection", annotated_frame)

                    # 如果检测到人物，进行跟踪
                    if banana_detected:
                        track_object(banana_x, banana_y, x_center, y_center)

                        # 检查目标是否在准心附近且未发射过
                        if (abs(banana_x - x_center) < 40 and abs(banana_y - y_center) < 30) and not fired:
                            ep_blaster.fire(fire_type=blaster.WATER_FIRE)
                            fired = True
                            print("发射!")
                else:
                    cv2.imshow("YOLOv8 Banana Detection", frame)
            # 按ESC退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        cv2.destroyAllWindows()
        ep_robot.close()


if __name__ == '__main__':
    main()