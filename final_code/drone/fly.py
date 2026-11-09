import cv2
import numpy as np
import time
import os
from djitellopy import Tello, TelloException

def force_land(drone) -> bool:
    """强制降落函数，增加重试次数和等待时间，返回是否成功降落"""
    print("尝试强制降落...")
    for i in range(5):  # 最多尝试5次
        try:
            drone.land()
            time.sleep(3)  # 等待降落动作执行
            print(f"降落尝试 {i + 1}/5 成功")
            return True
        except TelloException as te:
            print(f"降落尝试 {i + 1}/5 失败（Tello异常）: {te}，重试...")
            time.sleep(2)
        except Exception as e:
            print(f"降落尝试 {i + 1}/5 失败（其他异常）: {e}，重试...")
            time.sleep(2)
    print("警告：所有降落尝试失败，请手动干预！")
    return False

def main():
    # 创建保存照片的文件夹
    drone = Tello()
    emergency_landed = False
    # 照片计数器
    photo_counter = 0

    try:
        # 连接无人机
        print("正在连接无人机...")
        drone.connect()
        battery_percent = drone.get_battery()
        print(f"电池电量: {battery_percent}%")

        # 检查电池电量
        if battery_percent < 15:
            print("警告: 电池电量低于15%，请更换电池")
            return

        # 开启视频流
        print("开启摄像头...")
        drone.streamon()

        # 延长视频流初始化时间
        time.sleep(4)
        frame_read = drone.get_frame_read()

        # 检查初始帧是否有效（增加重试机制）
        frame_valid = False
        for _ in range(3):
            current_frame = frame_read.frame
            if current_frame is not None:
                frame_valid = True
                break
            print("无法获取视频流，重试...")
            drone.streamoff()
            time.sleep(2)
            drone.streamon()
            time.sleep(3)
            frame_read = drone.get_frame_read()
        if not frame_valid:
            print("视频流初始化失败，程序退出")
            force_land(drone)
            return

        # 获取帧尺寸（基于RGB转换后的画面）
        frame_bgr = frame_read.frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_width = frame_rgb.shape[1]
        frame_height = frame_rgb.shape[0]

        # 创建RGB专用显示窗口
        cv2.namedWindow("Drone View (RGB Format)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone View (RGB Format)", 960, 720)
        print("已开启RGB格式实时画面展示窗口")

        # 起飞并上升
        print("起飞...")
        drone.takeoff()
        time.sleep(2)
        
        # 拍摄第一张照片（RGB格式）
        print("拍摄第一张照片...")
        frame_bgr = frame_read.frame
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            photo_counter += 1
           
            cv2.imwrite("input/fly.jpg", frame_rgb)
            print(f"RGB照片已保存")
        else:
            print("无法获取有效帧，跳过拍照")
        time.sleep(2)

        # 后续任务：移动并拍摄
       
        # 任务完成，降落
        print("任务完成，准备降落...")
        if not emergency_landed:
            emergency_landed = force_land(drone)

    except TelloException as te:
        print(f"无人机命令错误: {str(te)}")
        if not emergency_landed:
            emergency_landed = force_land(drone)
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        if not emergency_landed:
            emergency_landed = force_land(drone)
    finally:
        # 清理资源
        print("关闭摄像头...")
        try:
            drone.streamoff()
        except:
            pass
        cv2.destroyAllWindows()
        try:
            drone.end()
        except:
            pass
        # 明确使用 emergency_landed 判断是否需要额外处理
        if emergency_landed:
            print("已执行紧急降落逻辑")
        print("程序结束，照片已保存至dji文件夹")


if __name__ == "__main__":
    main()
