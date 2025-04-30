# Cloud-tracking Development Log

## 25-04-29 updates

使用 [`main_inference_LaSOT_samurai.py`](./examples/inference/main_inference.py) 脚本对LaSOT数据集中部分数据进行推理，获得结果举例如下：

* airplane-6的[预测边界框信息](./examples/inference/LaSOT-results/samurai/samurai_base_plus/airplane-6.txt)及其[可视化](./examples/inference/LaSOT-visualization/samurai/base_plus/airplane-6.mp4)
* airplane-19的[预测边界框信息](/examples/inference/LaSOT-results/samurai/samurai_base_plus/airplane-19.txt)及其[可视化](./examples/inference/LaSOT-visualization/samurai/base_plus/airplane-19.mp4)

|             模型           | 服务器 | 单帧推理时长（ms） | 模型计算量 |
|----------------------------|--------|--------------------|------------|
|  sam2.1_hiera_base_plus.pt |  4090  |        83.2        |            |
|                            |  2080  |        88.0        |            |
|    sam2.1_hiera_large.pt   |  4090  |       106.3        |            |
|                            |  2080  |       134.8        |            |
|    sam2.1_hiera_small.pt   |  4090  |        72.5        |            |
|                            |  2080  |        70.5        |            |
|    sam2.1_hiera_tiny.pt    |  4090  |        71.2        |            |
|                            |  2080  |        66.5        |            |
