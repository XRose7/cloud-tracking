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
|    sam2.1_hiera_large.pt   |  4090  |        43.9        |            |
|                            |  2080  |       139.8        |            |
|    sam2.1_hiera_small.pt   |  4090  |        27.5        |            |
|                            |  2080  |        75.2        |            |
|    sam2.1_hiera_tiny.pt    |  4090  |        25.3        |            |
|                            |  2080  |        71.2        |            |

使用[`demo_two_gpu.py`](./scripts/demo_two_gpu.py)实现在不同gpu上加载不同模型进行推理，推理命令为：python demo_two_gpu.py --video1 video1.mp4 --model1 moedl1.pt -- out1 out1.mp4 --video2 video2.mp4 --model2 moedl2.pt --out2 out2.mp4。

conda list环境列表：[`environment.txt`](./environment.txt)
