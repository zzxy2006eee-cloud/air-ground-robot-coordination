import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class Correct():
    
    def __init__(self):
        self.point=[]
        self.points=[]
        
        self.ori_point=[]
        
        self.henxian=[]
        #红色范围
        self.lower_red_1 = np.array([0, 80, 80])
        self.upper_red_1 = np.array([10, 255, 255])
        self.lower_red_2 = np.array([170, 80, 80])
        self.upper_red_2 = np.array([180, 255, 255])
        
        #蓝色范围
        self.lower_blue=np.array([90,100,100])
        self.upper_blue=np.array([120,255,255])
        
        #蓝
        '''self.image_1=cv2.imread("input/bbbb.jpg")
        self.h,self.w=self.image_1.shape[:2]#获取高、宽
        self.lower_part=self.image_1[self.h//2:self.h,]#截掉上半部分
        self.mask_blue=self.blue_mask(self.lower_part)'''
        #红
        self.image_2=cv2.imread("input/fly.jpg")#读取无人机拍的照片
        self.h_r,self.w_r=self.image_2.shape[:2]#获取高、宽
        self.image_2=self.image_2[self.h_r//2:self.h_r,]#截掉上半部分,直接不要
        
        
        
        cv2.imwrite("input/ground.jpg",self.image_2)#这个作为后续处理的图,切掉上半部分，只保留地下
        
        self.img=self.increase('input/ground.jpg',sat_gain=1.0,val_gain=1.0)#校正后的图，坑较暗
        cv2.imwrite("input/ground_increase.jpg",self.img)#增加饱和度后的图
        
        
        self.image_2_r=cv2.imread("input/ground_increase.jpg")
        self.h_r,self.w_r=self.image_2_r.shape[:2]#获取高、宽
        self.cut=5
        self.lower_part_r=self.image_2_r[self.h_r//self.cut:self.h_r,]#截掉上面1/5
        self.mask_red=self.red_mask(self.lower_part_r)


    def red_mask(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#转换为HSV颜色空间
        mask_red_1=cv2.inRange(hsv,self.lower_red_1,self.upper_red_1)
        mask_red_2=cv2.inRange(hsv,self.lower_red_2,self.upper_red_2)
        red_=cv2.bitwise_or(mask_red_1,mask_red_2)
        return red_
    
    def blue_mask(self,image):#蓝色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#转换为HSV颜色空间
        blue=cv2.inRange(hsv,self.lower_blue,self.upper_blue)
        #去噪
        '''kernel=np.ones((5,5),np.uint8)
        mask=cv2.erode(mask,kernel,iterations=0)  #腐蚀
        mask=cv2.dilate(mask,kernel,iterations=1) #膨胀'''
        return blue

    def extract_color_pixels(self,mask):#目标颜色像素点
        #提取掩码中某颜色像素的坐标（x: 列，y: 行）
        y_coords, x_coords = np.where(mask == 255)  # np.where返回 (行索引, 列索引)，对应图像的 (y, x)
        if len(x_coords) == 0:
            raise ValueError(f"未检测到红颜色的像素")
        #整理为 (x, y) 格式的像素数组
        color_pixels = np.column_stack((x_coords, y_coords))
        return color_pixels#目标颜色像素点数组
    
    
    def lines(self, mask, eps=30, min_samples=200):
        """
        拟合图片中的多条直线
        参数:
            mask: 图像掩码
            eps: DBSCAN聚类的邻域半径
            min_samples: 形成密集区域所需的最小样本数
        返回:
            多条直线的端点列表，每条直线用两个端点表示
        """
        color_pixels = self.extract_color_pixels(mask)  # 目标颜色像素点
        
        if len(color_pixels) < 20:
            return []  # 像素点太少，无法拟合直线
        
        # 使用DBSCAN算法对像素点进行聚类，将属于同一直线的点聚集在一起
        # 可以根据实际数据调整eps和min_samples参数
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(color_pixels)
        labels = clustering.labels_
        
        # 获取不同的聚类（忽略噪声点，标签为-1）
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        lines = []
        # 对每个聚类分别拟合直线
        for label in unique_labels:
            # 获取当前聚类的所有点
            cluster_points = color_pixels[labels == label]
            
            # 如果聚类中的点太少，跳过
            if len(cluster_points) < 50:
                continue
                
            # 调用已有的单直线拟合方法
            line_endpoints = self.single_line(cluster_points)
           
    def save_points(self,point):#保存所需直线端点
        if len(self.point) < 4:
            self.point.append(point)
            if len(self.point) == 2:
                self.points.append(self.point[:])
            elif len(self.point) == 4:
                self.points.append(self.point[2:])
        
        
    def single_line(self, points):#拟合单条直线,基于原有的line方法修改
       
        x = points[:, 0]
        y = points[:, 1]
        n = len(x)
        
        # 计算斜率a和截距b（最小二乘法公式）
        numerator_a = n * np.sum(x * y) - np.sum(x) * np.sum(y)
        denominator_a = n * np.sum(x **2) - (np.sum(x))** 2
        
        if abs(denominator_a) < 1e-6:  # 分母接近0 → 垂直线（x = b）
            b = np.mean(x)  # 垂直线的x坐标取所有x的平均值
            a = float('inf')
            
            #垂直线的端点：取y的最小值和最大值
            min_y = np.min(y)
            max_y = np.max(y)
            p1 = (int(b), int(min_y))
            p2 = (int(b), int(max_y))
        else:  #普通直线（y = a*x + b）
            a = numerator_a / denominator_a  # 斜率
            b = (np.sum(y) - a * np.sum(x)) / n  # 截距
            
            #普通直线的端点：取x的最小值和最大值，计算对应y
            min_x = np.min(x)
            max_x = np.max(x)
            p1 = [int(min_x), int(a * min_x + b)] # 左端点
            p2 = [int(max_x), int(a * max_x + b)]  # 右端点
            
            p1=[p1[0],p1[1]+self.h_r//self.cut]
            p2=[p2[0],p2[1]+self.h_r//self.cut]
            
            if p1[1]<p2[1]:#将线延长
                p1_=-((b+self.h_r//self.cut)//a)
                p1_=[int(p1_),0]#与x轴交点
                p2_=p2
            else:
                p2_=-((b+self.h_r//self.cut)//a)
                p2_=[int(p2_),0]#与x轴交点
                p1_=p1
            
        
        
        if abs(a)>0.5:
            if p1[1]<p2[1]:#纵坐标小的在前面
                self.save_points(p1_)#保存端点,用于校正
                self.save_points(p2_)
            else:
                self.save_points(p2_)#大的在后面
                self.save_points(p1_)
            self.ori_point.append([p1,p2])
        else:
            #self.hen_points.append(a)
            self.henxian.append(p1)
            self.henxian.append(p2)
            
        a=cv2.line(self.image_2_r, p1, p2, (0, 0, 255), 2)
        cv2.imwrite("out_put/xian.jpg",a)
        return (p1, p2)

    def horizontal_correction(self,img, line_points):
        """水平校正函数"""
        # 计算线的角度
        x1, y1 = line_points[0]
        x2, y2 = line_points[1]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # 获取图像中心
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 执行旋转
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def correct_perspective(self,img, lines):#lines:线的两个端点     透视校正
        # 获取图像尺寸
        h, w = img.shape[:2]
        
        # 计算两条线的方向向量
        line1_vector = np.array(lines[0][1]) - np.array(lines[0][0])
        line2_vector = np.array(lines[1][1]) - np.array(lines[1][0])
        
        # 计算两条线的中点
        mid_point1 = (np.array(lines[0][0]) + np.array(lines[0][1])) / 2
        mid_point2 = (np.array(lines[1][0]) + np.array(lines[1][1])) / 2
        
        # 计算两条线的长度和角度
        line1_length = np.linalg.norm(line1_vector)
        line2_length = np.linalg.norm(line2_vector)
        
        # 计算线条的平均倾斜角度
        angle1 = np.arctan2(line1_vector[1], line1_vector[0])
        angle2 = np.arctan2(line2_vector[1], line2_vector[0])
        avg_angle = (angle1 + angle2) / 2
        
        # 计算源点（保持原始位置）
        src_points = np.float32([lines[0][0], lines[0][1], lines[1][0], lines[1][1]])
        
        # 计算目标点（保持相对位置，但确保垂直）
        # 保持线的长度和间距，只调整角度使其垂直
        x1, y1 = lines[0][0]
        x2, y2 = lines[1][0]
        
        if lines[0][0][0]>lines[1][0][0]:
            t=lines[0]
            lines[0]=lines[1]
            lines[1]=t
        src_points = np.float32([lines[0][0], lines[0][1], lines[1][0], lines[1][1]])
        #print(f"lll{lines}")
        # 计算校正后的点位置，保持线的长度但使其垂直
        dst_points = np.float32([
            [100,10],  #第一个点变到左上角
            [100, 3*h],  # 第一条线终点垂直向下
            [w-150,10],  # 第二条线起点变到右上角
            [w-150, 3*h]   # 第二条线终点垂直向下
        ])
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        
        self.m=M
        # 应用透视变换，保持原始色彩空间和质量
        corrected = cv2.warpPerspective(img, M, (w,4*h), 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,borderValue=(2,0,0))#黑色填充边界
        
        return corrected


    def m_(self,M,point):#[[,],[,]]
        point0=np.array([[[point[0][0],point[0][1]]]],dtype=np.float32)
        transformed_point=cv2.perspectiveTransform(point0,M)
        p1=[transformed_point[0][0][0],transformed_point[0][0][1]]
        
        point1=np.array([[[point[1][0],point[1][1]]]],dtype=np.float32)
        transformed_point_=cv2.perspectiveTransform(point1,M)
        p2=[transformed_point_[0][0][0],transformed_point_[0][0][1]]
       
       
        return([p1,p2])
    
    
    def increase(self,path='input/a_.jpg',sat_gain=1.5,val_gain=1.2):
        img=cv2.imread(path)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        s=np.clip(s*sat_gain,0,255).astype(np.uint8)#提高饱和度
        v=np.clip(v*val_gain,0,255).astype(np.uint8)#提高亮度
        enhanced_hsv=cv2.merge((h,s,v))
        enhanced_img=cv2.cvtColor(enhanced_hsv,cv2.COLOR_HSV2BGR)
        return enhanced_img
    
    def main(self):
        #--------------------------------------------------------------
        
        self.points=[]
        self.henxian=[]
        ori_r=cv2.imread("input/ground_increase.jpg")#作为原图
        
        cv2.imwrite("out_put/red_line.jpg",self.mask_red)#红线掩码
        
        self.lines(self.mask_red, eps=10, min_samples=5)#直线拟合
        points=self.points#用于校正的点
        print(points)
        
        corrected_image=self.correct_perspective(ori_r, points)#一次
        cv2.imwrite(f"input/a_0.jpg",corrected_image)#校正的图片
        corrected_image=self.horizontal_correction(corrected_image, self.henxian)#二次：水平
        cv2.imwrite(f"input/a2.jpg",corrected_image)#旋转的图片
       
        print(self.henxian)
        
        p=self.ori_point#竖线端点坐标----------------------------------((,) (,) (,) (,))
        
        print(f"初始竖线端点坐标{p}")
        
        h=self.henxian#横线端点坐标------------------------

        print(self.m)#M
        hen=self.m_(self.m,h)#横线端点转换后
        #print(hen_p)
        shu1=self.m_(self.m,p[0])
        shu2=self.m_(self.m,p[1])
        print(f"第一条竖线的点：{shu1}。第二条竖线的点:{shu2}")
        line_length0 = np.sqrt((shu1[1][0]- shu1[0][0]) ** 2 + (shu1[1][1] - shu1[0][1]) ** 2)#像素长度
        line_length1 = np.sqrt((shu2[1][0]- shu2[0][0]) ** 2 + (shu2[1][1] - shu2[0][1]) ** 2)#像素长度
        a=(line_length0+line_length1)//2#竖线
        b=np.sqrt((hen[1][0]-hen[0][0]) ** 2 + (hen[1][1] -hen[0][1]) ** 2)#横线
        b=int(b)
        self.k=b/a#一个比值
        print(f"k={self.k}")
        img=cv2.imread("input/a_0.jpg")
        h_img, w_img= img.shape[:2]
        print(f"图像长宽：{h_img},{w_img}")

        target_w=int(w_img)
        target_h=int(h_img*self.k)
        resized_img=cv2.resize(img,(target_w,target_h),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("input/a_.jpg",resized_img)#完全校正的图片
        
        pixels_to_m=b/0.6
        with open("out_put/pixels_to_cm.txt","w") as f:
                f.write(str(pixels_to_m))#像素与实际长度比值cm
        
if __name__ == "__main__":
    correct=Correct()
    correct.main()