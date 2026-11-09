#将拼接图片的颜色提取出来,识别起点终点
import cv2
import numpy as np
import math
class Image_process():
    def __init__(self):
        
        self.img=self.increase('input/a_.jpg',sat_gain=2,val_gain=1.0)#校正后的图，坑较暗
        cv2.imwrite("input/a.jpg",self.img)#增加饱和度后的图
        
        self.image_path='input/a.jpg'
        self.image = cv2.imread(self.image_path)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)#转换为HSV颜色空间
        #黑色范围
        self.lower_black=np.array([0,0,0])    #HSV下限 色调:0红 60绿 120蓝  饱和度:0纯灰   明度:0黑 255白
        self.upper_black=np.array([180,255,140]) #HSV上限
        #蓝色范围
        self.lower_blue=np.array([90,100,100])
        self.upper_blue=np.array([120,255,255])
        #红色范围
        self.lower_red_1 = np.array([0, 80, 50])
        self.upper_red_1 = np.array([20, 255, 255])
        self.lower_red_2 = np.array([160, 80, 50])
        self.upper_red_2 = np.array([180, 255, 255])
        
        
    def mask_black(self):#黑色
        mask=cv2.inRange(self.hsv,self.lower_black,self.upper_black)
        #去噪
        kernel=np.ones((5,5),np.uint8)
        mask=cv2.erode(mask,kernel,iterations=0)  #腐蚀去除小噪点
        mask=cv2.dilate(mask,kernel,iterations=6) #膨胀填充空洞
        return mask
    
    def mask_blue(self):#蓝色
        mask=cv2.inRange(self.hsv,self.lower_blue,self.upper_blue)
        #去噪
        kernel=np.ones((5,5),np.uint8)
        mask=cv2.erode(mask,kernel,iterations=0)  #腐蚀去除小噪点
        mask=cv2.dilate(mask,kernel,iterations=7) #膨胀填充空洞
        return mask
        
    def mask_red(self):#红色
        mask_red_1=cv2.inRange(self.hsv,self.lower_red_1,self.upper_red_1)
        mask_red_2=cv2.inRange(self.hsv,self.lower_red_2,self.upper_red_2)
        red_mask=cv2.bitwise_or(mask_red_1,mask_red_2)
        #去噪
        kernel=np.ones((5,5),np.uint8)
        red_mask=cv2.erode(red_mask,kernel,iterations=1)  # 腐蚀去除小噪点
        red_mask=cv2.dilate(red_mask,kernel,iterations=8) # 膨胀填充空洞
        return red_mask
    
    def only_red(self,hsv):#红
        mask_red_1=cv2.inRange(hsv,self.lower_red_1,self.upper_red_1)
        mask_red_2=cv2.inRange(hsv,self.lower_red_2,self.upper_red_2)
        red_mask=cv2.bitwise_or(mask_red_1,mask_red_2)
        return red_mask
    
    def only_blue(self,hsv):#蓝色
        mask=cv2.inRange(hsv,self.lower_blue,self.upper_blue)
        return mask
    
    def increase(self,path='input/a_.jpg',sat_gain=1.5,val_gain=1.2):
        img=cv2.imread(path)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        s=np.clip(s*sat_gain,0,255).astype(np.uint8)#提高饱和度
        v=np.clip(v*val_gain,0,255).astype(np.uint8)#提高亮度
        enhanced_hsv=cv2.merge((h,s,v))
        enhanced_img=cv2.cvtColor(enhanced_hsv,cv2.COLOR_HSV2BGR)
        return enhanced_img
    
    def start_end(self,mask):#圆形起点或终点
        #识别圆形起点 param min_area: 最小面积阈值 param circularity_threshold: 圆形度阈值(0-1，1为理想圆形)
        
        # 2. 提取轮廓（只检测外轮廓，压缩水平、垂直和对角线方向的元素）
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL,  # 只检测外轮廓
            cv2.CHAIN_APPROX_SIMPLE  # 轮廓点压缩
        )
        valid_contours = []
        for contour in contours:
            # 3. 计算面积并筛选
            area = cv2.contourArea(contour)
            if area < 5:#面积
                continue  #面积太小，跳过
            
            # 4.计算周长和圆形度
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter == 0:
                continue  #避免除零错误
            # 圆形度公式：4π×面积 / 周长²（理想圆形为1，值越接近1越圆）
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity >= 0.4:#圆形度阈值(0-1，1为理想圆形)
                valid_contours.append((contour, area))
        # 5. 检查是否有符合条件的区域
        if not valid_contours:
            raise ValueError(f"未找到面积大于100且圆形度符合要求的红色区域")
        # 按面积排序，选择最大的圆形区域
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        largest_contour = valid_contours[0][0]
        # 计算该区域的质心作为起始点
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            raise ValueError("无法计算轮廓质心")
        
        # 质心坐标 (x, y)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        return (cX, cY)

    def detect_color_object(self):#将原本图像的蓝红黑部分重新表示出来
        #创建蓝红黑色掩膜
        
        self.blue=self.mask_blue()
        self.red=self.mask_red()
        self.black=self.mask_black()
        #变换颜色
        self.color_mask=np.zeros_like(self.image)
        
        self.color_mask[self.black>0]=(0,100,0)#黑色->绿色
        self.color_mask[self.blue>0]=(254,1,1)#蓝色->蓝色
        self.color_mask[self.red>0]=(1,1,250)#红色->红色
        
        cv2.imwrite("out_put/image_with_color_mask_.jpg",self.color_mask)#保存结果
    
if __name__ == "__main__":
    to_process=Image_process()
    
    to_process.detect_color_object()
    
    img=cv2.imread('out_put/image_with_color_mask_.jpg') 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转换为HSV颜色空间
    mask=to_process.only_red(hsv)
    a=to_process.start_end(mask)
    print(a)
    a_c=cv2.circle(img, a, 3, (0, 255, 255), -1)
    cv2.imwrite('a__.jpg',a_c)
    
    mask2=to_process.only_blue(hsv)
    b=to_process.start_end(mask2)
    print(b)
    b_c=cv2.circle(img, b, 6, (0, 255, 255), -1)
    cv2.imwrite('b__.jpg',a_c)