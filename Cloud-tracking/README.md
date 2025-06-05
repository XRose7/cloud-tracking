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

使用[`benchmark.py`](./examples/Batch-timing/benchmark.py)脚本实现批量统计各模型对单帧图像推理用时，[`test.mp4`](./examples/inference/test.mp4)以及[`test.txt`](./examples/inference/test.txt)为处理的视频及目标首帧位置，命令：python benchmark.py，统计结果如下表：

|          硬件         |   模型   |1024| sam2（ms）|samurai（ms）|480| sam2（ms）|samurai（ms）|
|-----------------------|---------|----|-----------|-------------|---|-----------|-------------|
|          4090​​	      |  tiny   |    |    28.0   |     29.4    |   |       |         |
|                       |  small  |    |    29.6   |     30.3    |   |       |         |
|                       |  large  |    |    44.6   |     41.8    |   |       |         |
|                       |  base   |    |    34.6   |     31.7    |   |       |         |
|          2080​​	      |  tiny   |    |    65.2   |     67.1    |   |    30.2   |    31.8     |
|                       |  small  |    |    70.1   |     70.0    |   |    32.7   |    36.0     |
|                       |  large  |    |   136.4   |    140.1    |   |    53.3   |    55.6     |
|                       |  base   |    |    87.0   |     88.2    |   |    38.0   |    39.9     |
|NVIDIA Jetson Orin​​ 64GB​​|  tiny   |    |   309.9   |             |   |       |         |
|                       |  small  |    |   302.3   |             |   |       |         |
|                       |  large  |    |   676.0   |             |   |       |         |
|                       |  base   |    |   488.6   |             |   |       |         |

使用[`inference.py`](./examples/inference/inference.py)中注释掉的determine_model_cfg()函数进行sam2推理用时的统计，命令为python ./scripts/inference.py。

Jetson特殊配置: 
torch需要官网下载对应版本的whl进行pip安装;
decord需要clone git repo到本地build

使用[`inference_on_two_gpus.py`](./examples/inference/inference_on_two_gpus.py)实现在不同gpu上加载不同模型进行推理，命令为：python ./scripts/inference_on_two_gpu.py --video1 video1.mp4 --model1 moedl1.pt -- out1 out1.mp4 --video2 video2.mp4 --model2 moedl2.pt --out2 out2.mp4。

## 25-05-12 updates

使用prometheus + grafana可视化模型性能。

[`visualization_example.py`](./examples/visualization/visualization_example.py)：对虚拟机上0-100随机数（模拟帧率）以及CPU占用率监测，结果见[visualization.mp4](./examples/visualization/visualization.mp4)，后续只需在[`inference.py`](./examples/inference/inference.py)的for循环内部暴露帧率等指标即可，命令为python visualization_examples.py。

Prometheus-3.4.0 + grafana-9.3.2，解压后在对应目录下通过以下命令运行，详见[visualization_deploy.txt](./visualization/visualization_deploy.txt)：

./prometheus --config.file=prometheus.yml

./bin/grafana-server

## 25-06-05 updates

服务端、客户端：pip install grpcio grpcio-tools

通过gRPC，将本地客户端的待处理视频、目标首帧位置、处理命令发送至服务器端，服务器处理后视频保存至客户端。

服务端：cd [scripts/](./scripts)

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. demo.proto，
将生成的demo_dp2.py及demo_dp2_grpc.py保存至客户端[scripts/](./scripts)目录下

python ['./scripts/server.py'](./scripts/server.py)启动 gRPC 服务

客户端：

python ./scripts/inference_with_grpc_and_visualization.py通过 gRPC 调用远程服务器执行推理逻辑，并实现可视化。

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
       
requirements.txt 可能包含了本项目不需要的很多依赖，建议只加必须项（√）

README里面
cd checkpoints 应该是 cd sam2/checkpoints/（√）


