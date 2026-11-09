# 运行指南

## 离线检查与规划

在仓库根目录执行 `python tools/verify_archive.py`，检查源码语法和原始文件的 SHA-256。这个检查不代表算法正确性或硬件兼容性验证。

安装 `requirements.txt` 后执行：

```powershell
python tools/run_planning.py --image assets/drone_capture.jpg
```

每次会建立独立的 `runtime/<时间>/input` 和 `runtime/<时间>/out_put`，把输入准备成原程序要求的 `input/fly.jpg` 后运行原始 `All.py`。可用 `--output runtime/my_run` 指定新的空目录。失败时保留中间图供排查，退出码为 1；不会把已有运行结果覆盖掉。当前这张样例会在标定直线提取后报错，命令用于复现和调试，不是已经跑通的全流程演示。

查看单文件图像规划实验，可在桌面图形环境中执行：

```powershell
cd experiments/planning
python image_planner.py
```

该实验使用同目录下的 `road mini.jpg` 并打开结果窗口，按任意键结束。本次该样例正常结束但未绘制出黄色路径；它在无路时不报错，因此窗口出现不等于规划成功。它也不生成小车需要的完整角度/距离文件。

若手动运行 `All.py`，必须先在工作目录建立 `input/`、`out_put/` 并放入 `input/fly.jpg`。Python 脚本位置与工作目录不同是允许的，原程序中的相对路径均以工作目录为准。

## 数据文件与单位

| 文件（运行目录内） | 含义 | 下游用途 |
| --- | --- | --- |
| `input/fly.jpg` | 无人机前视采图 | `Correct` 输入 |
| `input/a_.jpg` | 透视及比例校正后图像 | 颜色分割输入 |
| `input/a.jpg` | 饱和度增强图 | 路径绘制背景 |
| `out_put/image_with_color_mask_.jpg` | 障碍、起点和终点颜色图 | BFS 输入 |
| `out_put/smooth_path_points.txt` | 最终路径点，每行 `x, y`，单位为像素 | 角度和距离计算 |
| `out_put/pixels_to_cm.txt` | 名称沿用原文件；实际为像素/米 | 距离换算 |
| `out_put/angles.txt` | 每段转向角，单位度 | 小车转向 |
| `out_put/path_distance.txt` | 每段位移，单位米 | 小车前进 |

`line_test.py` 用标定线像素长度除以 `0.6`，因此比例值实际是像素/米。`angle.py` 再用像素距离除以此比例，得到米。不要因文件名含 `cm` 再做一次厘米换算。

## 无人机与地面机器人

在独立的硬件环境安装 `requirements-hardware.txt`，按实际设备和 SDK 配置网络。仓库没有原项目的依赖锁定文件；两种无人机入口分别使用 `djitellopy.Tello` 与 `robomaster.robot.Drone`，应按设备选用，不要连续执行两份脚本。

无人机脚本写入工作目录的 `input/fly.jpg`，需要先建立 `input/`。`drone.py` 在模块顶层连接、起飞和拍照；不要通过导入它来做环境检查。

`car_move.py` 从工作目录读取 `angles.txt` 与 `path_distance.txt`，而规划器把它们写在 `out_put/`。原始项目没有自动搬运脚本，硬件运行前需把已检查的两份文件复制到小车工作目录，两者行数应一致。

YOLO 使用 `yolov8n.pt`。仓库不重复上传权重，按所用 Ultralytics 环境准备该文件并确认来源。小车源码包含固定的初始位移、转向以及 `WATER_FIRE` 执行器调用；属于原结题演示行为，使用前须按实际场地及教学设备审核。此次整理未执行这些硬件脚本。

## 精选实验

实验保留原始逻辑和固定参数，英文名称只用于便于检索，不表示它们已被修复或已通过硬件验证。

| 实验 | 建议工作目录 | 备注 |
| --- | --- | --- |
| `planning/image_planner.py` | `experiments/planning` | 自带 `road mini.jpg`；会打开 OpenCV 窗口 |
| `planning/coordinate_planner.py` | `experiments/planning` | 自带 `road 5.jpg`；输出存在已知数组判断问题 |
| `stitching/orb_stitch.py` | `experiments/stitching` | 自带两张拼接输入图；匹配和融合质量未复测 |
| `vision/webcam_yolo.py` | 自选工作目录 | 电脑摄像头 0、YOLOv8 权重 |
| `vision/hsv_color_detection.py` | 自选工作目录 | 电脑摄像头颜色识别 |
| `vision/robomaster_phone.py`、`robomaster_banana.py` | 自选工作目录 | 需要 RoboMaster；原型含执行器逻辑 |
| `drone/manual_capture.py` | 自选工作目录 | 手动按 A 采图，原脚本不执行起飞 |
| `drone/red_marker_alignment.py`、`s_pattern_search.py` | 自选工作目录 | 需要无人机及对应场地标记，包含运动指令 |

历史 `processing/image_stitch.py` 由原作者标注“不用了”，保留作拼接研究参考，未被 `All.py` 调用；使用时另备 `images/` 和 `middle_process_images/`。
