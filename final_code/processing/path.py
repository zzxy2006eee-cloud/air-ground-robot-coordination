#找出并标出黄色路径,保存坐标点，完成
import cv2
import numpy as np
from collections import deque
from file import *
from math import sqrt
from image_processing import Image_process
import time
#------------------------------------------------原始路径，没有标点
class Path():
    
    def __init__(self):
        self.file=File()
        to_process=Image_process()
        
        img=cv2.imread('out_put/image_with_color_mask_.jpg') 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转换为HSV颜色空间
        mask_start=to_process.only_red(hsv)
        mask_end=to_process.only_blue(hsv)
        self.start=to_process.start_end(mask_start)
        self.end=to_process.start_end(mask_end)
        self.ori_image=cv2.imread('input/a.jpg')
        
        #绿色范围
        self.lower_green=np.array([35,43,46])
        self.upper_green=np.array([77,255,255])
        self.image=cv2.imread('out_put/image_with_color_mask_.jpg') 
        self.hsv=cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #障碍物掩码 #二值数组 obstacle_mask[y, x] == 0表示非障碍区
        self.obstacle_mask=cv2.inRange(self.hsv, self.lower_green,self.upper_green)
    
    # Ramer - Douglas - Peucker算法
    def rdp(self,points, epsilon):#points:输入的点集，epsilon:阈值，值越大，简化后的点越少.
        dmax = 0#记录最大垂直距离
        index = 0#记录最大距离对应的点的索引
        end = len(points) - 1#最后一个点的索引
        for i in range(1, end):
            d = self.perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        if dmax > epsilon:
            results = self.rdp(points[:index + 1], epsilon)[:-1] + self.rdp(points[index:], epsilon)
        else:
            results = [points[0], points[end]]
        return results#返回简化后的点集

    # 计算点到直线的垂直距离
    def perpendicular_distance(self,point, start, end):
        if start == end:
            return sqrt((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2)
        else:
            n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
            d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            return n / d

    def find_path(self,image):
        ori_image=cv2.imread('input/a.jpg')
        height, width = image.shape[:2]
    # 定义方向数组
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 广度优先搜索
        queue = deque([(self.start, 0)])  # 队列中存储 (当前位置, 路径长度)
        visited = set([self.start])
        parent = {}
        found = False
        while queue:
            current, distance = queue.popleft()
            if current ==self.end:
                found = True
                print("找到终点")
                break
            for dx, dy in directions:
                new_x, new_y = current[0] + dx, current[1] + dy
                if 0 <= new_x < width and 0 <= new_y < height \
                        and self.obstacle_mask[new_y, new_x] == 0 and (new_x, new_y) not in visited:#0表示非障碍区
                    queue.append(((new_x, new_y), distance + 1))
                    visited.add((new_x, new_y))
                    parent[(new_x, new_y)] = current
        if found:# 绘制最短路径
            print("found path")
            path = []#路径中所有的点
            current = self.end
            while current !=self.start:
                path.append(current)
                current = parent[current]
            path.append(self.start)
            path.reverse()
            for point in path:
                cv2.circle(ori_image, point, 3, (0, 255, 255), -1)  #在每个点画半径为3的⚪形成黄色路径
        #保存结果
        cv2.imwrite('out_put/image_with_path.jpg',ori_image)
        return path#返回坐标点数组

    def smooth_path(self,epsilon,path_points):
        ori_image=cv2.imread('input/a.jpg')
        # 使用RDP算法平滑路径
        # epsilon可以调整这个值来控制平滑程度
        smoothed_path = self.rdp(path_points, epsilon)
        
        #绘制平滑后的路
        
        # 绘制连接点的路径（线条）
        for i in range(1, len(smoothed_path)):
            cv2.line(ori_image, smoothed_path[i-1], smoothed_path[i], (0, 255, 255), 6)
        for point in smoothed_path:
            cv2.circle(ori_image, point, 3, (0, 0, 255), -1)  #转折点
        # 保存结果
        #cv2.imshow('Rasterized Image with Path',ori_image)
        cv2.imwrite('out_put/image_with_smoothed_path.jpg',ori_image)
        print(smoothed_path)
        return smoothed_path#返回光滑处理的坐标点的数组
    def move_path(self,ori_points,p):
        points=[]
        for (w,h) in ori_points:
            num=self.way(w,h,3)
            a=0
            if num==0:
                for y in range(h,h-p,-1):#上面一定范围有障碍物
                    
                    if self.obstacle_mask[y,w]==255:
                        points.append((w,y+p+50))
                        a=1
                        print('范围')
                        break
                for y in range(h,h+p):
                    if self.obstacle_mask[y,w]==255:
                        points.append((w,y-p-50))
                        a=1
                        print('范围')
                        break
                if a==0:
                    points.append((w,h))
            elif num==1:
                points.append((w,h+p))
            elif num==2:
                points.append((w,h-p))
            elif num==4:
                points.append((w+p,h))
            elif num==7:
                points.append((w-p,h))
            elif num==5 or num==10:
                points.append((w+p,h+p))
            elif num==8 or num==12:
                points.append((w-p,h+p))
            elif num==6 or num==11:
                points.append((w+p,h-p))
            elif num==9 or num==13:
                points.append((w-p,h-p))
        
        # 绘制连接点的路径（线条）
        for i in range(1, len(points)):
            cv2.line(self.ori_image,points[i-1],points[i], (0,255,255),6)
        for point in points:
            cv2.circle(self.ori_image,point,3,(0,0,255),-1)  #转折点
        cv2.imwrite('out_put/moved_path.jpg',self.ori_image)
        
        return points
            
            
    def way(self,w,h,d):#判断障碍物在那个方向
        mask=self.obstacle_mask
        t=0
        p=0
        if mask[h-d,w]==255:#上方有障碍物 ↑  
            t+=1
        if mask[h+d,w]==255:#下  ↓
            t+=2
        if mask[h,w-d]==255:#左  ← 
            t+=4
        if mask[h,w+d]==255:#右  →
            t+=7
            
        if (t==0 or t==1 or t==2 or t==4 or t==7):#仅一个方向有障碍物，判断斜向，是否处于拐角边缘
            if mask[h-d,w-d]==255:#左上  ↖*
                p+=1
            if mask[h+d,w-d]==255:#左下  ↙*
                p+=1
            if mask[h-d,w+d]==255:#右上  ↗*
                p+=1
            if mask[h+d,w+d]==255:#右下  ↘*
                p+=1
        if p==1:
            if mask[h-d,w-d]==255:#左上  ↖*
                t=10
            if mask[h+d,w-d]==255:#左下  ↙*
                t=11
            if mask[h-d,w+d]==255:#右上  ↗*
                t=12
            if mask[h+d,w+d]==255:#右下  ↘*
                t=13
        return t


if __name__ == "__main__":
    time1=time.time()
    path=Path()
    file=File()
    image=cv2.imread('out_put/image_with_color_mask_.jpg')
    path_points=path.find_path(image)
    file.save_points('out_put/ori_path_points.txt',path_points)

    smoothed_points=path.smooth_path(25,path_points)
    file.save_points('out_put/smooth_path_points.txt',smoothed_points)
    
    a=path.move_path(smoothed_points,100)
    
    '''smoothed_points=path.smooth_path(30,a)
    file.save_points('smooth_path_points.txt',smoothed_points)
   '''
    time2=time.time()
    c=time2-time1
    #print(f"规划用时{c}秒")