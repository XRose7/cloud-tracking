import grpc
from concurrent import futures
import demo_pb2
import demo_pb2_grpc

import torch
import numpy as np

from demo import main as run_demo, determine_model_cfg, prepare_frames_or_path, load_txt
from sam2.build_sam import build_sam2_video_predictor

class SamService(demo_pb2_grpc.SamServiceServicer):
    def RunDemo(self, request, context):
        print(f"Received request: {request}")

        model_cfg = determine_model_cfg(request.model_path)
        predictor = build_sam2_video_predictor(model_cfg, request.model_path, device="cuda")

        frames_or_path = prepare_frames_or_path(request.video_path)
        prompts = load_txt(request.txt_path)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
            bbox, track_label = prompts[0]
            _, _, _ = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            for frame_idx, object_ids, masks, maskmem_features, maskmem_pos_enc in predictor.propagate_in_video(state):
                # flatten tensor to list
                # features_list = maskmem_features.cpu().numpy().flatten().tolist()
                # print("Dtype of maskmem_features:", maskmem_features.dtype)
                # print(maskmem_features.shape)
                # print(maskmem_pos_enc[0].shape)
                features_list = maskmem_features.to(dtype=torch.float32).cpu().numpy().flatten().tolist()
                
                # flatten pos_enc list of tensors
                pos_enc_all = []
                for t in maskmem_pos_enc:
                    if isinstance(t, torch.Tensor):
                        pos_enc_all.extend(t.cpu().numpy().flatten().tolist())

                yield demo_pb2.DemoResponse(
                    frame_idx=frame_idx,
                    maskmem_features=features_list,
                    maskmem_pos_enc=pos_enc_all,
                    maskmem_features_shape=list(maskmem_features.shape)
                )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_SamServiceServicer_to_server(SamService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at port 50051.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
