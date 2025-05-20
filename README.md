# Cloud-tracking Development Log

## 25-04-29 updates

使用 [`main_inference_LaSOT_samurai.py`](./examples/inference/main_inference.py) 脚本对LaSOT数据集中部分数据进行推理，获得结果举例如下：

* airplane-6的[预测边界框信息](./examples/inference/LaSOT-results/samurai/samurai_base_plus/airplane-6.txt)及其[可视化](./examples/inference/LaSOT-visualization/samurai/base_plus/airplane-6.mp4)
* airplane-19的[预测边界框信息](/examples/inference/LaSOT-results/samurai/samurai_base_plus/airplane-19.txt)及其[可视化](./examples/inference/LaSOT-visualization/samurai/base_plus/airplane-19.mp4)

## 25-05-06 updates

### Download Checkpoints
First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

cd sam2/checkpoints/ && \
./download_ckpts.sh && \
cd ..

使用[`benchmark.py`](./benchmark.py)脚本实现批量统计各模型对单帧图像推理用时，[`test.mp4`](./test.mp4)以及[`test.txt`](./test.txt)为处理的视频及目标首帧位置，命令：python benchmark.py，统计结果如下表：

|          硬件         |   模型   | sam2（ms）|samurai（ms）|
|-----------------------|---------|-----------|-------------|
|          4090​​	      |  tiny   |    28.0   |     29.4    |
|                       |  small  |    29.6   |     30.3    |
|                       |  large  |    44.6   |     41.8    |
|                       |  base   |    34.6   |     31.7    |
|          2080​​	      |  tiny   |    65.2   |     67.1    |
|                       |  small  |    70.1   |     70.0    |
|                       |  large  |   136.4   |    140.1    |
|                       |  base   |    87.0   |     88.2    |
|NVIDIA Jetson Orin​​ 64GB​​|  tiny   |   309.9   |         |
|                       |  small  |   302.3   |         |
|                       |  large  |   676.0   |         |
|                       |  base   |   488.6   |         |

使用[`demo.py`](./scripts/demo.py)中注释掉的determine_model_cfg()函数进行sam2推理用时的统计。

Jetson特殊配置: 
torch需要官网下载对应版本的whl进行pip安装;
decord需要clone git repo到本地build

使用[`demo_two_gpu.py`](./scripts/demo_two_gpu.py)实现在不同gpu上加载不同模型进行推理，命令为：python demo_two_gpu.py --video1 video1.mp4 --model1 moedl1.pt -- out1 out1.mp4 --video2 video2.mp4 --model2 moedl2.pt --out2 out2.mp4。

## 25-05-12 updates

使用prometheus + grafana可视化模型性能。

[`monitor_example.py`](./monitor_example.py)：对虚拟机上0-100随机数（模拟帧率）以及CPU占用率监测，结果见[监测视频.MP4](./监测视频.mp4)，后续只需在[`demo.py`](./scripts/demo.py)的for循环内部暴露帧率等指标即可。

Prometheus-3.4.0 + grafana-9.3.2，解压后在对应目录下通过以下命令运行：

./prometheus --config.file=prometheus.yml

./bin/grafana-server

# 环境：

conda env export > environment.yml：[environment.yml](./environment.yml)

pip freeze > requirements.txt: [requirements.txt](./requirements.txt)

## 一些建议：

benchmark.py
video_path txt_path使用相对路径（√）

缺少 import sys（√）
最好加上检查
   if result.returncode != 0:
       print(result.stderr)
       
requirements.txt 可能包含了本项目不需要的很多依赖，建议只加必须项

README里面
cd checkpoints 应该是 cd sam2/checkpoints/（√）


