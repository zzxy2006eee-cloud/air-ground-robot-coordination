import cv2
import time
import numpy as np


def detect_color_object(frame, lower_color, upper_color):
    """识别并标记特定颜色的物体（默认检测蓝色）"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 形态学去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 查找轮廓并绘制框
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def main():
    # 定义颜色范围（默认蓝色）
    lower_color = np.array([90, 50, 50])
    upper_color = np.array([130, 255, 255])

    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0为默认摄像头，可改为视频文件路径
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    last_process_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取画面！")
                break

            # 每隔0.01秒处理一次
            current_time = time.time()
            if current_time - last_process_time >= 0.01:
                processed_frame = detect_color_object(frame.copy(), lower_color, upper_color)
                cv2.imshow("Processed Frame", processed_frame)
                last_process_time = current_time

            # 实时显示原始画面
            cv2.imshow("Camera Feed", frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()