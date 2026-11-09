from robomaster import robot
from robomaster import camera
import time
import cv2


tl_drone=robot.Drone()
tl_drone.initialize()

tl_camera=tl_drone.camera
battery =tl_drone.battery
tl_flight=tl_drone.flight

current_battery =battery.get_battery()
print('电池电量：', current_battery, '%')

tl_camera.start_video_stream(display=True)
time.sleep(3)
tl_flight.takeoff().wait_for_completed()
time.sleep(5)
img =tl_camera.read_cv2_image(strategy='newest')
cv2.imwrite("input/fly.jpg", img)
print("保存")
time.sleep(2)
tl_flight.land().wait_for_completed()

tl_camera.stop_video_stream()
tl_drone.close()