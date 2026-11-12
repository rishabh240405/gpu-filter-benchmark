# GPU Image Filter Benchmark

GPU Specialization capstone project. The program applies a Gaussian blur to a
batch of large images with four implementations and compares speed and
accuracy:

| Method                      | Description                                                         |
|-----------------------------|---------------------------------------------------------------------|
| `cpu_single_thread`         | Reference implementation, also used to check GPU correctness        |
| `cuda_global_memory`        | CUDA kernel, one thread per pixel, all reads from global memory     |
| `cuda_shared_memory`        | CUDA kernel with shared-memory tiling and a halo of `radius` pixels |
| `cuda_shared_streams_N`     | Shared-memory kernel with pinned memory and N CUDA streams so that copies and kernels of different images overlap |

The filter weights are stored in constant memory. Borders are handled by
clamping coordinates. Kernel time is measured with CUDA events, total time
(including host/device copies) with a wall-clock timer.

## Requirements
- NVIDIA GPU and the CUDA toolkit (`nvcc`), C++17 compiler
- Python 3 with `matplotlib` and `pillow` (only for the plots)
- Developed and tested on: **[fill in, e.g. Google Colab, Tesla T4]**

No dataset needs to be downloaded: the program generates deterministic
synthetic images (gradient + bright disc + noise) in memory. The default
batch is 20 images of 2048x2048 pixels (about 84 million pixels).

## Build and run

    make
    ./gpu_filter_benchmark.exe

or build, run and save the log in one step:

    ./run.sh

Example with custom arguments:

    ./run.sh --num_images 30 --width 4096 --height 4096 --radius 7 --streams 8

## Command line arguments

| Argument        | Meaning                                        | Default                       |
|-----------------|------------------------------------------------|-------------------------------|
| `--num_images`  | Number of images in the batch                  | 20                            |
| `--width`       | Image width in pixels                          | 2048                          |
| `--height`      | Image height in pixels                         | 2048                          |
| `--radius`      | Gaussian filter radius (1-15)                  | 3                             |
| `--streams`     | CUDA streams for the overlapped version        | 4                             |
| `--save_images` | 1 = save first input/output image as PGM       | 1                             |
| `--output_dir`  | Directory for saved images                     | `output`                      |
| `--csv_path`    | Result CSV file                                | `output/benchmark_results.csv`|
| `--help`        | Print usage                                    |                               |

## Experiments and plots

    ./scripts/run_experiments.sh      # radius sweep and stream-count sweep
    python3 scripts/plot_results.py   # writes charts and before/after image to plots/

## Output files
- `output/execution_log.txt` - console log of the main run
- `output/benchmark_results.csv` - timings, speedup and max error per method
- `output/input_image0.pgm`, `output/output_blurred_image0.pgm` - first image before and after
- `results/*.csv`, `results/log_*.txt` - experiment sweeps
- `plots/*.png` - charts and the before/after comparison

## Repository layout

    src/main.cu                  CUDA program (kernels, host code, CLI)
    Makefile, run.sh             build and run helpers
    scripts/run_experiments.sh   parameter sweeps
    scripts/plot_results.py      charts and before/after image
    docs/PROJECT_DESCRIPTION.md  written description of the project
    output/, results/, plots/    proof-of-execution artifacts

## Running on Google Colab
1. Runtime -> Change runtime type -> T4 GPU.
2. Upload or clone this repository, then run `!make` and `!./run.sh`.
