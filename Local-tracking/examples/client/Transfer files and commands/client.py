import grpc
import demo_pb2
import demo_pb2_grpc

def generate_chunks(filepaths):
    for path in filepaths:
        filename = path.split("/")[-1]
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield demo_pb2.VideoChunk(filename=filename, content=chunk)

def save_result(response_iterator, output_path):
    with open(output_path, "wb") as f:
        for chunk in response_iterator:
            f.write(chunk.content)

def run():
    with grpc.insecure_channel("10.1.114.151:50051") as channel:
        stub = demo_pb2_grpc.VideoProcessorStub(channel)
        video_path = "../examples/inference/test.mp4"
        txt_path = "../examples/inference/test.txt"
        responses = stub.ProcessVideo(generate_chunks([video_path, txt_path]))
        save_result(responses, "../examples/inference/client_result.mp4")
        print("处理完成，结果保存在 ../examples/inference/client_result.mp4")

if __name__ == "__main__":
    run()
