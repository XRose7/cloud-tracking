# Cloud-tracking Development Log

## 25-04-29 updates

使用 [`main_inference_LaSOT_samurai.py`](./examples/inference/main_inference.py) 脚本对LaSOT数据集中部分数据进行推理，获得结果举例如下：

* airplane-6的[预测边界框信息](./examples/inference/LaSOT-results/samurai/samurai_base_plus/airplane-6.txt)及其[可视化](./examples/inference/LaSOT-visualization/samurai/base_plus/airplane-6.mp4)
* airplane-19的[预测边界框信息](/examples/inference/LaSOT-results/samurai/samurai_base_plus/airplane-19.txt)及其[可视化](./examples/inference/LaSOT-visualization/samurai/base_plus/airplane-19.mp4)

## 25-05-06 updates

### Download Checkpoints
First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

cd checkpoints && \
./download_ckpts.sh && \
cd ..


使用[`benchmark.py`](./benchmark.py)脚本实现批量统计各模型对单帧图像推理用时，[`mouse-8.mp4`](./mouse-8.mp4)以及[`test.txt`](./test.txt)为处理的视频及目标首帧位置。

|             模型           | 服务器 | 单帧推理时长（ms） | 模型计算量 |
|----------------------------|--------|--------------------|------------|
|  sam2.1_hiera_base_plus.pt |  4090  |        32.4        |            |
|                            |  2080  |        90.8        |            |
|                            |  NVIDIA Jetson Orin 64GB  |        488.6        |            |
|    sam2.1_hiera_large.pt   |  4090  |        43.9        |            |
|                            |  2080  |       139.8        |            |
|                            |  NVIDIA Jetson Orin 64GB  |        676.0        |            |
|    sam2.1_hiera_small.pt   |  4090  |        27.5        |            |
|                            |  2080  |        75.2        |            |
|                            |  NVIDIA Jetson Orin 64GB  |        302.3        |            |
|    sam2.1_hiera_tiny.pt    |  4090  |        25.3        |            |
|                            |  2080  |        71.2        |            |
|                            |  NVIDIA Jetson Orin 64GB  |        309.9        |            |

 
Jetson特殊配置: 
torch需要官网下载对应版本的whl进行pip安装;
decord需要clone git repo到本地build

一些建议：
benchmark.py
video_path txt_path使用相对路径
缺少 import sys
最好加上检查
   if result.returncode != 0:
       print(result.stderr)
requirements.txt 可能包含了本项目不需要的很多依赖，建议只加必须项
README里面
cd checkpoints 应该是 cd sam2/checkpoints/




使用[`demo_two_gpu.py`](./scripts/demo_two_gpu.py)实现在不同gpu上加载不同模型进行推理，推理命令为：python demo_two_gpu.py --video1 video1.mp4 --model1 moedl1.pt -- out1 out1.mp4 --video2 video2.mp4 --model2 moedl2.pt --out2 out2.mp4。

conda env export > environment.yml：[environment.yml](./environment.yml)

pip freeze > requirements.txt: [requirements.txt](./requirements.txt)

## 25-05-12 updates

使用prometheus + grafana可视化模型性能。

[`monitor_example.py`](./monitor_example.py)：对虚拟机上0-100随机数（模拟帧率）以及CPU占用率监测，结果见[监测视频.MP4](./监测视频.mp4)，后续只需在[`demo.py`](./scripts/demo.py)的for循环内部暴露帧率等指标即可。
