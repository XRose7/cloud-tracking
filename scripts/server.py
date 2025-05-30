import grpc
from concurrent import futures
import demo_pb2
import demo_pb2_grpc
import os

class VideoProcessorServicer(demo_pb2_grpc.VideoProcessorServicer):
    def ProcessVideo(self, request_iterator, context):
        input_dir = "server_input"
        output_path = "server_output/output.mp4"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs("server_output", exist_ok=True)

        files = {}
        for chunk in request_iterator:
            filepath = os.path.join(input_dir, chunk.filename)
            if chunk.filename not in files:
                files[chunk.filename] = open(filepath, "wb")
            files[chunk.filename].write(chunk.content)
        
        for f in files.values():
            f.close()

        # Run demo.py
        video_path = os.path.join(input_dir, "../examples/inference/test.mp4")
        txt_path = os.path.join(input_dir, "../examples/inference/test.txt")
        cmd = f"conda run -n samurai python demo.py --video_path {video_path} --txt_path {txt_path} --video_output_path {output_path}"
        os.system(cmd)

        # Stream back result
        with open(output_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield demo_pb2.VideoChunk(filename="result.mp4", content=chunk)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_VideoProcessorServicer_to_server(VideoProcessorServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started at port 50051.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
