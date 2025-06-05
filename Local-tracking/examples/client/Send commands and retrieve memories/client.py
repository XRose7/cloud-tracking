import grpc
import demo_pb2
import demo_pb2_grpc

import numpy as np
import os
import pickle

def run():
    channel = grpc.insecure_channel('10.1.114.128:50051')
    stub = demo_pb2_grpc.SamServiceStub(channel)

    request = demo_pb2.DemoRequest(
        video_path="./test.mp4",
        txt_path="./test.txt",
        model_path="./sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    )

    result_dict = {}

    for response in stub.RunDemo(request):
        print(f"Received frame {response.frame_idx}")
        result_dict[response.frame_idx] = {
            "maskmem_features": np.array(response.maskmem_features, dtype=np.float32),
            "maskmem_pos_enc": np.array(response.maskmem_pos_enc, dtype=np.float32)
        }

    # 保存为 pickle 或 npy 文件
    os.makedirs("output", exist_ok=True)
    with open("output/maskmem_features_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    print("Saved maskmem features and pos_enc to output/maskmem_features_dict.pkl")

if __name__ == '__main__':
    run()
