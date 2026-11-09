from ultralytics import YOLO
import cv2

# 加载 YOLOv8 模型
model = YOLO('yolov8n.pt')  # 可以根据需要选择不同的预训练模型，如 yolov8s.pt, yolov8m.pt 等

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头读取一帧图像
    ret, frame = cap.read()

    # 使用 YOLOv8 进行目标检测
    results = model(frame)[0]

    # 遍历检测到的目标
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        # 转换坐标为整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 获取类别名称
        class_name = model.names[int(class_id)]

        # 在图像上绘制边界框和类别标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Webcam Object Detection', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()