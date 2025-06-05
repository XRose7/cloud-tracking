import threading
import time
import grpc
import demo_pb2
import demo_pb2_grpc
import numpy as np
import os
import pickle
import argparse
import torch
import gc
import cv2
import sys
import os.path as osp
import pynvml
from prometheus_client import start_http_server, Gauge
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

# 本地记忆
result_dict = {}

# Prometheus 指标
frame_processing_time = Gauge(
    'frame_processing_time_seconds',
    'Time taken to process each frame'
)
average_processing_time = Gauge(
    'average_processing_time_seconds',
    'Average time taken per frame'
)
gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'Current GPU utilization percentage'
)

def run_client():
    print("[Client Thread] Starting gRPC client...")
    channel = grpc.insecure_channel('10.1.114.128:50051')
    stub = demo_pb2_grpc.SamServiceStub(channel)

    request = demo_pb2.DemoRequest(
        video_path="./test.mp4",
        txt_path="./test.txt",
        model_path="./sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    )


    for response in stub.RunDemo(request):
        # print(f"[Client Thread] Received frame {response.frame_idx}")
        result_dict[response.frame_idx] = {
            "maskmem_features": np.array(response.maskmem_features, dtype=np.float32),
            "maskmem_pos_enc": np.array(response.maskmem_pos_enc, dtype=np.float32)
        }
        result_dict["original_shape"] = list(response.maskmem_features_shape)
        
    os.makedirs("output", exist_ok=True)
    with open("output/maskmem_features_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    print("[Client Thread] Done and saved to output/maskmem_features_dict.pkl")


def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def visualization_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认使用第一个 GPU
    while True:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization.set(util.gpu)
        time.sleep(5)  # 每 5 秒更新一次

def run_demo(model_path, video_path, txt_path, video_output_path, save_to_video):
    # 启动 Prometheus 指标服务器
    start_http_server(8000)
    
    # 启动 GPU 监控线程
    gpu_thread = threading.Thread(target=visualization_gpu_utilization)
    gpu_thread.daemon = True
    gpu_thread.start() 
    
    print("[Demo Thread] Starting local demo inference...")
    model_path = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    video_path = "./examples/inference/test.mp4"
    txt_path = "./examples/inference/test.txt"
    video_output_path = "demo.mp4"
    save_to_video = True

    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda")
    frames_or_path = prepare_frames_or_path(video_path)
    prompts = load_txt(txt_path)

    if osp.isdir(video_path):
        frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg")])
        loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
        height, width = loaded_frames[0].shape[:2]
    else:
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()
        height, width = loaded_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        predictor.update_memory_from_grpc(result_dict)
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            predictor.update_memory_from_grpc(result_dict)
            
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            # 单帧处理时间
            frame_processing_time.set(predictor.per_time)
            
            # 平均处理时间
            if predictor.count_frame > 0:
                avg_time = predictor.all_time / predictor.count_frame
                average_processing_time.set(avg_time)
                
            if save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = (255, 0, 0)
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

                out.write(img)
            
        print(f"[Demo Thread] AVERAGE TIME: {predictor.all_time / predictor.count_frame}s")
        out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    
    model_path = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    video_path = "./examples/inference/test.mp4"
    txt_path = "./examples/inference/test.txt"
    video_output_path = "demo.mp4"
    save_to_video = True
    
    t1 = threading.Thread(target=run_client)
    t2 = threading.Thread(
        target=run_demo,
        args=(model_path, video_path, txt_path, video_output_path, save_to_video)
    )
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("[Main] Both threads finished.")
