#小车转向角度和运动距离

import math
from file import File

class Angle():
    @staticmethod
    def get_angle(datas):
        angles=[]
        v1=(1,0)#初始向量
        for i in range(len(datas)-1):
            v2=(datas[i+1][0]-datas[i][0],datas[i+1][1]-datas[i][1])#第二段向量
            
            dot_product=v1[0]*v2[0]+v1[1]*v2[1]#点积
            mag_v1=math.sqrt(v1[0]**2+v1[1]**2)#向量模长
            mag_v2=math.sqrt(v2[0]**2+v2[1]**2)
            
            cos=dot_product/(mag_v1*mag_v2)#余弦值
            cos=max(min(cos,1.0),-1.0)#误差修正
            
            angle_rad = math.acos(cos)#弧度
            angle_deg = math.degrees(angle_rad)#角度
            angle_deg=int(angle_deg)
            
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]#叉积判断方向
            if cross_product < 0:
                angle_deg = -angle_deg
                
            #if abs(angle_deg)<=10:
                #angle_deg=0
            angles.append(angle_deg)
            t=v2
            v1=t
        with open("out_put/angles.txt","w") as f:
            for i in angles:
                f.write(str(i)+"\n")
        return angles
    
    @staticmethod
    def distance(points):
        l=[]
        with open("out_put/pixels_to_cm.txt") as f:
            num=f.read()
            num=float(num)
            
        for i in range(len(points)-1):
            v=(points[i+1][0]-points[i][0],points[i+1][1]-points[i][1])
            mag_v=math.sqrt(v[0]**2+v[1]**2)#模长
            #mag_v=int(mag_v)
            
            real_len=mag_v/num
            real_len=round(real_len,2)
            l.append(real_len)  
            
        with open("out_put/path_distance.txt","w") as f:
            for i in l:
                f.write(str(i)+"\n")
        return l#m
            
        
if __name__=="__main__":
    points=File.read_points('out_put/smooth_path_points.txt')
    angles=Angle.get_angle(points)
    print(f"各个转向角度：{angles}")
    
    d=Angle.distance(points)
    print(f"各段距离：{d}")
    