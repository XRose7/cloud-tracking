import subprocess
import re
import sys


def main():
    # paths to use in the benchmark
    video_path = "../examples/inference/test.mp4"
    txt_path = "../examples/inference/test.txt"

    # list of model checkpoint files to test
    model_paths = [
        "sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        "sam2/checkpoints/sam2.1_hiera_tiny.pt",
        "sam2/checkpoints/sam2.1_hiera_small.pt",
        "sam2/checkpoints/sam2.1_hiera_large.pt",
    ]

    # dictionary to hold average times per model
    average_times = {}

    for idx, model_path in enumerate(model_paths, 1):
        run_times = []
        for run_idx in range(1, 6):
            print(f"Processing Model {idx}/{len(model_paths)}: '{model_path}'  -  Run {run_idx}/5")
            # print(video_path)
            # print(txt_path)

            # invoke the demo script and capture its output
            result = subprocess.run(
                [
                    "python", "demo.py",
                    "--video_path", video_path,
                    "--txt_path", txt_path,
                    "--model_path", model_path,
                    # you can disable video writing to speed up if desired:
                    # "--save_to_video", "False"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
              print("Subprocess failed:")
              print(result.stderr)
              sys.exit(1)
	     
            # parse the printed average time
            match = re.search(r"AVERAGE:\s*([0-9\.]+)s", result.stdout)
            if not match:
                print("Failed to parse average time from demo output:")
                print(result.stdout)
                sys.exit(1)

            elapsed = float(match.group(1))
            run_times.append(elapsed)
            print(f"  -> Run {run_idx} time: {elapsed:.4f}s")

        # compute the average for this model
        avg_time = sum(run_times) / len(run_times)
        average_times[model_path] = avg_time
        print(f"Average time for '{model_path}': {avg_time:.4f}s\n")

    # display summary table
    print("Summary of average processing times:")
    print(f"{'Model':<50} {'Avg Time (s)':>15}")
    print('-' * 65)
    for model, avg in average_times.items():
        print(f"{model:<50} {avg:>15.4f}")


if __name__ == "__main__":
    main()
