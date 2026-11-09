#保存坐标点，读取坐标点,从文件夹读取图片
import cv2
import os
class File():
    
    @staticmethod
    def read_points(path):#读取坐标点
        points=[]
        with open(path, "r") as file:
            # 逐行读取文件
            for line in file:
                line = line.strip()
                if line:  # 跳过空行
                    # 按逗号分割，得到x和y字符串
                    x_str, y_str = line.split(",")
                    # 转换为整数（或浮点数，根据需求）并添加到数组
                    points.append((int(x_str.strip()), int(y_str.strip())))
        return points

    @staticmethod
    def save_points(filename,points):#保存坐标点
        """将路径点保存到文件中"""
        try:
            with open(filename, 'w') as f:
                for point in points:
                    f.write(f"{point[0]}, {point[1]}\n")
            print(f"路径已成功保存到 {filename}")
        except Exception as e:
            print(f"保存路径时出错: {e}")
    
    
    @staticmethod
    def read_images(dir, extensions=("jpg", "png")):#读取文件夹多张图片
        images = []
        for filename in os.listdir(dir):
            if filename.lower().endswith(extensions):
                file_path = os.path.join(dir, filename)
                img = cv2.imread(file_path)
                if img is not None:
                    images.append(img)
        return images