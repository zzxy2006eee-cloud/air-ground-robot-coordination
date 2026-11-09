#正常  无人机拍照后处理
import cv2
import numpy as np
import math
from file import File
from image_processing import Image_process
from path import Path
#from line import Line
from angle import Angle
from line_test import Correct

file=File()

image_oir=cv2.imread('input/fly.jpg')#读取无人机拍的原始图片

correct=Correct()
correct.main()#校正后的图片 "input/a.jpg"

to_process=Image_process()
to_process.detect_color_object()#颜色覆盖
color_mask=cv2.imread('out_put/image_with_color_mask_.jpg') 


path=Path()
path_points=path.find_path(color_mask)#保存图片out_put/image_with_path.jpg    返回初始路径点
file.save_points('out_put/ori_path_points.txt',path_points)#保存路径坐标点

#先移点再平滑
ori_points=file.read_points('out_put/ori_path_points.txt')#读取路径坐标点
smoothed_points=path.smooth_path(5,ori_points)#路径平滑
file.save_points('out_put/path_points.txt',smoothed_points)#提取出几个转折点

ori_points=file.read_points('out_put/path_points.txt')
moved_points=path.move_path(ori_points,100)#移动点
file.save_points('out_put/moved_path_points.txt',moved_points)

ori_points_2=file.read_points('out_put/moved_path_points.txt')#读取路径坐标点
smoothed_points=path.smooth_path(40,ori_points_2)#路径平滑         第二次
file.save_points('out_put/smooth_path_points.txt',smoothed_points)#保存平滑处理后的坐标点

points=File.read_points('out_put/smooth_path_points.txt')

angles=Angle.get_angle(points)#保存各个转向角度#获取并保存行驶距离与转向角度
print(f"各个转向角度：{angles}")
d=Angle.distance(points)#保存各段距离
print(f"各段距离：{d}")
