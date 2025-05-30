import argparse
import os
import os.path as osp
import threading
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    prompts = {}
    for fid, line in enumerate(lines):
        x, y, w, h = map(float, line.split(','))
        prompts[fid] = ((int(x), int(y), int(x + w), int(y + h)), 0)
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
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of frames.")

def run_worker(video_path, txt_path, model_path, video_output_path, gpu_id, worker_name):

    device = f"cuda:{gpu_id}"
    print(f"[{worker_name}] Starting on {device}")

    cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(cfg, model_path, device=device)

    frames_or_path = prepare_frames_or_path(video_path)
    prompts = load_txt(txt_path)

    frame_rate = 30
    if True:
        if osp.isdir(video_path):
            frame_files = sorted([osp.join(video_path, f)
                                  for f in os.listdir(video_path)
                                  if f.lower().endswith((".jpg", ".jpeg"))])
            loaded_frames = [cv2.imread(p) for p in frame_files]
            h, w = loaded_frames[0].shape[:2]
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
            if not loaded_frames:
                raise RuntimeError(f"[{worker_name}] No frames loaded from {video_path}")
            h, w = loaded_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_output_path, fourcc, frame_rate, (w, h))

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)

        bbox0, _ = prompts[0]
        predictor.add_new_points_or_box(state, box=bbox0, frame_idx=0, obj_id=0)

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            img = loaded_frames[frame_idx]
            for obj_id, mask in zip(obj_ids, masks):
                m = mask[0].cpu().numpy() > 0.0
                mask_img = np.zeros_like(img)
                mask_img[m] = color[(obj_id+1) % len(color)]
                img = cv2.addWeighted(img, 1.0, mask_img, 0.2, 0)
                ys, xs = np.where(m)
                if ys.size:
                    x0, y0 = xs.min(), ys.min()
                    x1, y1 = xs.max(), ys.max()
                    cv2.rectangle(img, (x0,y0), (x1,y1), color[obj_id % len(color)], 2)
            writer.write(img)

    avg_time = predictor.all_time / predictor.count_frame
    print(f"[{worker_name}] Avg time/frame: {avg_time:.3f}s")
    print("\n")

    writer.release()
    del predictor, state
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{worker_name}] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", required=True)
    parser.add_argument("--txt1",   required=True)
    parser.add_argument("--model1", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--out1",   default="out1.mp4")
    parser.add_argument("--video2", required=True)
    parser.add_argument("--txt2",   required=True)
    parser.add_argument("--model2", default="sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--out2",   default="out2.mp4")
    args = parser.parse_args()

    t1 = threading.Thread(
        target=run_worker,
        kwargs=dict(
            video_path=args.video1,
            txt_path=args.txt1,
            model_path=args.model1,
            video_output_path=args.out1,
            gpu_id=0,
            worker_name="Worker-0"
        )
    )
    t2 = threading.Thread(
        target=run_worker,
        kwargs=dict(
            video_path=args.video2,
            txt_path=args.txt2,
            model_path=args.model2,
            video_output_path=args.out2,
            gpu_id=1,
            worker_name="Worker-1"
        )
    )

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("All workers finished.")
