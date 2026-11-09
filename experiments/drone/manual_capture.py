import cv2
import os
import time
from djitellopy import Tello, TelloException


def main():
    # 确保dji文件夹存在
    if not os.path.exists("dji"):
        os.makedirs("dji")
        print("已创建dji文件夹用于保存照片")
    else:
        print("使用已存在的dji文件夹保存照片")

    drone = Tello()
    try:
        # 连接无人机
        print("正在连接无人机...")
        drone.connect()
        print(f"无人机连接成功，电池电量: {drone.get_battery()}%")

        # 检查电池电量
        if drone.get_battery() < 30:
            print("警告: 电池电量低于30%，建议及时充电")

        # 开启视频流（不执行起飞操作）
        print("开启无人机摄像头（无需起飞）...")
        drone.streamon()

        # 等待视频流初始化稳定
        time.sleep(5)
        frame_read = drone.get_frame_read()

        # 照片计数器
        photo_counter = 0

        print("操作提示：")
        print("  - 按A键拍摄照片（保存至dji文件夹）")
        print("  - 按Q键停止程序")

        # 主循环（持续运行直到手动停止）
        while True:
            # 获取无人机实时画面
            frame = frame_read.frame
            if frame is None:
                print("暂无法获取画面，重试中...")
                time.sleep(0.5)
                continue

            # 转换为RGB格式（仅保留此转换后的画面）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 直接使用RGB格式画面进行显示（需要转换为BGR才能被OpenCV正确显示）
            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 显示状态信息
            cv2.putText(display_frame,
                        f"Photos taken: {photo_counter}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(display_frame,
                        f"Battery: {drone.get_battery()}%",
                        (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(display_frame,
                        "RGB Converted View | Handheld Mode",
                        (10, display_frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

            # 仅显示转换颜色通道后的画面
            cv2.imshow("Drone Camera (RGB Converted)", display_frame)

            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF

            # 按A键拍照（保存RGB格式）
            if key == ord('a') or key == ord('A'):
                photo_counter += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                photo_path = os.path.join("dji", f"rgb_photo_{timestamp}_{photo_counter}.jpg")

                # 保存转换后的RGB格式照片
                cv2.imwrite(photo_path, rgb_frame)
                print(f"RGB照片已保存：{photo_path}")

            # 按Q键手动停止程序
            elif key == ord('q') or key == ord('Q'):
                print("收到退出指令，正在关闭程序...")
                break

    except TelloException as e:
        print(f"无人机指令错误: {str(e)}")
    except Exception as e:
        print(f"程序错误: {str(e)}")
    finally:
        # 清理资源
        print("关闭摄像头...")
        try:
            drone.streamoff()  # 关闭视频流
        except:
            pass
        cv2.destroyAllWindows()  # 关闭所有显示窗口
        try:
            drone.end()  # 结束无人机连接
        except:
            pass
        print("程序已停止")


if __name__ == "__main__":
    main()
