// Copyright 2026. All rights reserved.
//
// GPU image-filter benchmark. Applies a Gaussian blur to a batch of large
// images with four implementations and compares them:
//   1. CPU single-thread reference
//   2. CUDA kernel reading from global memory
//   3. CUDA kernel with shared-memory tiling
//   4. Shared-memory kernel with pinned memory and multiple CUDA streams
//      (copies and kernels of different images overlap)
//
// Usage: gpu_filter_benchmark.exe [options]   (see --help)

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kMaxRadius = 15;
constexpr int kMaxFilterWidth = 2 * kMaxRadius + 1;
constexpr int kTile = 16;
constexpr int kMaxTileWidth = kTile + 2 * kMaxRadius;

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t error_code = (call);                                     \
    if (error_code != cudaSuccess) {                                     \
      std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                   \
                   cudaGetErrorString(error_code), __FILE__, __LINE__);  \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

// The filter weights live in constant memory, which is cached and broadcast
// to all threads of a warp that read the same address.
__constant__ float kFilter[kMaxFilterWidth * kMaxFilterWidth];

struct Options {
  int num_images = 20;
  int width = 2048;
  int height = 2048;
  int radius = 3;
  int num_streams = 4;
  int save_images = 1;
  bool show_help = false;
  std::string output_dir = "output";
  std::string csv_path = "output/benchmark_results.csv";
};

struct Result {
  std::string name;
  double total_ms;   // End-to-end time including host/device copies.
  double kernel_ms;  // Kernel-only time, or negative if not measured.
  double max_error;  // Maximum absolute difference from the CPU result.
};

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ int ClampDevice(int value, int size) {
  return value < 0 ? 0 : (value >= size ? size - 1 : value);
}

// Baseline: every thread reads all of its neighbours from global memory.
__global__ void BlurGlobalKernel(const float* input, float* output, int width,
                                 int height, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int filter_width = 2 * radius + 1;
  float sum = 0.0f;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int py = ClampDevice(y + dy, height);
    for (int dx = -radius; dx <= radius; ++dx) {
      const int px = ClampDevice(x + dx, width);
      sum += kFilter[(dy + radius) * filter_width + (dx + radius)] *
             input[py * width + px];
    }
  }
  output[y * width + x] = sum;
}

// Optimized: the block first loads its tile (plus a halo of `radius` pixels)
// into shared memory, so each input pixel is read from global memory once per
// block instead of once per neighbouring thread.
__global__ void BlurSharedKernel(const float* input, float* output, int width,
                                 int height, int radius) {
  __shared__ float tile[kMaxTileWidth][kMaxTileWidth];

  const int tile_width = kTile + 2 * radius;
  const int base_x = blockIdx.x * kTile - radius;
  const int base_y = blockIdx.y * kTile - radius;

  for (int ty = threadIdx.y; ty < tile_width; ty += kTile) {
    for (int tx = threadIdx.x; tx < tile_width; tx += kTile) {
      const int gx = ClampDevice(base_x + tx, width);
      const int gy = ClampDevice(base_y + ty, height);
      tile[ty][tx] = input[gy * width + gx];
    }
  }
  __syncthreads();

  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
  if (x >= width || y >= height) return;

  const int filter_width = 2 * radius + 1;
  float sum = 0.0f;
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      sum += kFilter[(dy + radius) * filter_width + (dx + radius)] *
             tile[threadIdx.y + radius + dy][threadIdx.x + radius + dx];
    }
  }
  output[y * width + x] = sum;
}

void LaunchBlur(bool use_shared, const float* input, float* output, int width,
                int height, int radius, cudaStream_t stream) {
  const dim3 block(kTile, kTile);
  const dim3 grid((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  if (use_shared) {
    BlurSharedKernel<<<grid, block, 0, stream>>>(input, output, width, height,
                                                 radius);
  } else {
    BlurGlobalKernel<<<grid, block, 0, stream>>>(input, output, width, height,
                                                 radius);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

void PrintUsage() {
  std::printf(
      "Usage: gpu_filter_benchmark.exe [options]\n"
      "  --num_images N   number of images in the batch (default 20)\n"
      "  --width W        image width in pixels (default 2048)\n"
      "  --height H       image height in pixels (default 2048)\n"
      "  --radius R       Gaussian filter radius, 1-%d (default 3)\n"
      "  --streams S      CUDA streams for the overlapped version (default 4)\n"
      "  --save_images B  1 = save first input/output image as PGM (default 1)\n"
      "  --output_dir D   directory for saved images (default output)\n"
      "  --csv_path P     result CSV file (default output/benchmark_results.csv)\n"
      "  --help           show this message\n",
      kMaxRadius);
}

bool ParseInt(const char* text, int min_value, int max_value, int* out) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < min_value || value > max_value) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool ParseArgs(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    bool ok = false;
    if (arg == "--help") {
      options->show_help = true;
      return true;
    } else if (arg == "--num_images" && has_value) {
      ok = ParseInt(argv[++i], 1, 100000, &options->num_images);
    } else if (arg == "--width" && has_value) {
      ok = ParseInt(argv[++i], 16, 16384, &options->width);
    } else if (arg == "--height" && has_value) {
      ok = ParseInt(argv[++i], 16, 16384, &options->height);
    } else if (arg == "--radius" && has_value) {
      ok = ParseInt(argv[++i], 1, kMaxRadius, &options->radius);
    } else if (arg == "--streams" && has_value) {
      ok = ParseInt(argv[++i], 1, 32, &options->num_streams);
    } else if (arg == "--save_images" && has_value) {
      ok = ParseInt(argv[++i], 0, 1, &options->save_images);
    } else if (arg == "--output_dir" && has_value) {
      options->output_dir = argv[++i];
      ok = true;
    } else if (arg == "--csv_path" && has_value) {
      options->csv_path = argv[++i];
      ok = true;
    }
    if (!ok) {
      std::fprintf(stderr, "Invalid or unknown argument: %s\n", arg.c_str());
      PrintUsage();
      return false;
    }
  }
  return true;
}

// Creates a deterministic test image: a gradient, a bright disc and noise.
void GenerateImage(float* image, int width, int height, int seed) {
  std::mt19937 rng(static_cast<unsigned int>(seed) + 12345u);
  std::uniform_real_distribution<float> noise(-25.0f, 25.0f);
  const float center_x = width * (0.25f + 0.5f * ((seed * 37) % 100) / 100.0f);
  const float center_y = height * (0.25f + 0.5f * ((seed * 53) % 100) / 100.0f);
  const float disc_radius = 0.2f * static_cast<float>(std::min(width, height));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float value = 127.0f * static_cast<float>(x) / static_cast<float>(width);
      const float dx = x - center_x;
      const float dy = y - center_y;
      if (dx * dx + dy * dy < disc_radius * disc_radius) value += 110.0f;
      value += noise(rng);
      image[static_cast<size_t>(y) * width + x] =
          std::min(255.0f, std::max(0.0f, value));
    }
  }
}

std::vector<float> MakeGaussianFilter(int radius) {
  const int filter_width = 2 * radius + 1;
  const float sigma = std::max(0.5f * static_cast<float>(radius), 0.5f);
  std::vector<float> filter(static_cast<size_t>(filter_width) * filter_width);
  float sum = 0.0f;
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      const float value =
          std::exp(-static_cast<float>(x * x + y * y) / (2.0f * sigma * sigma));
      filter[(y + radius) * filter_width + (x + radius)] = value;
      sum += value;
    }
  }
  for (float& value : filter) value /= sum;
  return filter;
}

int ClampHost(int value, int size) {
  return value < 0 ? 0 : (value >= size ? size - 1 : value);
}

void BlurCpu(const float* input, float* output, int width, int height,
             int radius, const std::vector<float>& filter) {
  const int filter_width = 2 * radius + 1;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int dy = -radius; dy <= radius; ++dy) {
        const int py = ClampHost(y + dy, height);
        for (int dx = -radius; dx <= radius; ++dx) {
          const int px = ClampHost(x + dx, width);
          sum += filter[(dy + radius) * filter_width + (dx + radius)] *
                 input[static_cast<size_t>(py) * width + px];
        }
      }
      output[static_cast<size_t>(y) * width + x] = sum;
    }
  }
}

double MaxAbsError(const float* a, const float* b, size_t count) {
  double max_error = 0.0;
  for (size_t i = 0; i < count; ++i) {
    max_error = std::max(max_error, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return max_error;
}

bool WritePgm(const std::string& path, const float* image, int width,
              int height) {
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  file << "P5\n" << width << " " << height << "\n255\n";
  std::vector<unsigned char> bytes(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<unsigned char>(
        std::min(255.0f, std::max(0.0f, image[i] + 0.5f)));
  }
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  return file.good();
}

// ---------------------------------------------------------------------------
// The four implementations
// ---------------------------------------------------------------------------

Result RunCpu(const Options& options, const std::vector<float>& filter,
              const float* host_in, float* reference) {
  const size_t pixels = static_cast<size_t>(options.width) * options.height;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < options.num_images; ++i) {
    BlurCpu(host_in + i * pixels, reference + i * pixels, options.width,
            options.height, options.radius, filter);
  }
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - start;
  return {"cpu_single_thread", elapsed.count(), -1.0, 0.0};
}

// Processes images one at a time on the default stream:
// copy in, kernel, copy out. Kernel time is measured with CUDA events.
Result RunGpuSequential(const std::string& name, bool use_shared,
                        const Options& options, const float* host_in,
                        float* host_out, const float* reference) {
  const size_t pixels = static_cast<size_t>(options.width) * options.height;
  const size_t image_bytes = pixels * sizeof(float);
  float* device_in = nullptr;
  float* device_out = nullptr;
  CUDA_CHECK(cudaMalloc(&device_in, image_bytes));
  CUDA_CHECK(cudaMalloc(&device_out, image_bytes));
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  // Warm-up run so one-time driver start-up cost is not timed.
  CUDA_CHECK(cudaMemcpy(device_in, host_in, image_bytes,
                        cudaMemcpyHostToDevice));
  LaunchBlur(use_shared, device_in, device_out, options.width, options.height,
             options.radius, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::memset(host_out, 0, pixels * options.num_images * sizeof(float));

  double kernel_ms = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < options.num_images; ++i) {
    CUDA_CHECK(cudaMemcpy(device_in, host_in + i * pixels, image_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    LaunchBlur(use_shared, device_in, device_out, options.width,
               options.height, options.radius, 0);
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaMemcpy(host_out + i * pixels, device_out, image_bytes,
                          cudaMemcpyDeviceToHost));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    kernel_ms += milliseconds;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - start;

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaFree(device_in));
  CUDA_CHECK(cudaFree(device_out));
  return {name, elapsed.count(), kernel_ms,
          MaxAbsError(host_out, reference, pixels * options.num_images)};
}

// Round-robins images over several streams so that the copy of one image
// overlaps with the kernel of another. Requires pinned host memory.
Result RunGpuStreams(const Options& options, const float* host_in,
                     float* host_out, const float* reference) {
  const size_t pixels = static_cast<size_t>(options.width) * options.height;
  const size_t image_bytes = pixels * sizeof(float);
  const int stream_count = std::min(options.num_streams, options.num_images);

  std::vector<cudaStream_t> streams(stream_count);
  std::vector<float*> device_in(stream_count, nullptr);
  std::vector<float*> device_out(stream_count, nullptr);
  for (int s = 0; s < stream_count; ++s) {
    CUDA_CHECK(cudaStreamCreate(&streams[s]));
    CUDA_CHECK(cudaMalloc(&device_in[s], image_bytes));
    CUDA_CHECK(cudaMalloc(&device_out[s], image_bytes));
  }
  std::memset(host_out, 0, pixels * options.num_images * sizeof(float));
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < options.num_images; ++i) {
    const int s = i % stream_count;
    CUDA_CHECK(cudaMemcpyAsync(device_in[s], host_in + i * pixels, image_bytes,
                               cudaMemcpyHostToDevice, streams[s]));
    LaunchBlur(true, device_in[s], device_out[s], options.width,
               options.height, options.radius, streams[s]);
    CUDA_CHECK(cudaMemcpyAsync(host_out + i * pixels, device_out[s],
                               image_bytes, cudaMemcpyDeviceToHost,
                               streams[s]));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - start;

  for (int s = 0; s < stream_count; ++s) {
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
    CUDA_CHECK(cudaFree(device_in[s]));
    CUDA_CHECK(cudaFree(device_out[s]));
  }
  return {"cuda_shared_streams_" + std::to_string(stream_count),
          elapsed.count(), -1.0,
          MaxAbsError(host_out, reference, pixels * options.num_images)};
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void PrintResults(const std::vector<Result>& results, int num_images) {
  const double cpu_ms = results.front().total_ms;
  std::printf("\n%-28s %12s %12s %12s %10s %12s\n", "method", "total_ms",
              "kernel_ms", "ms/image", "speedup", "max_error");
  for (const Result& r : results) {
    char kernel_text[32];
    if (r.kernel_ms < 0.0) {
      std::snprintf(kernel_text, sizeof(kernel_text), "NA");
    } else {
      std::snprintf(kernel_text, sizeof(kernel_text), "%.3f", r.kernel_ms);
    }
    std::printf("%-28s %12.3f %12s %12.3f %9.2fx %12.6f\n", r.name.c_str(),
                r.total_ms, kernel_text, r.total_ms / num_images,
                cpu_ms / r.total_ms, r.max_error);
  }
}

bool WriteCsv(const std::string& path, const std::vector<Result>& results,
              const Options& options) {
  const std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream file(path);
  if (!file) return false;
  const double cpu_ms = results.front().total_ms;
  file << "method,num_images,width,height,radius,total_ms,kernel_ms,"
          "ms_per_image,speedup_vs_cpu,max_abs_error\n";
  for (const Result& r : results) {
    file << r.name << "," << options.num_images << "," << options.width << ","
         << options.height << "," << options.radius << "," << r.total_ms << ",";
    if (r.kernel_ms < 0.0) {
      file << "NA";
    } else {
      file << r.kernel_ms;
    }
    file << "," << r.total_ms / options.num_images << ","
         << cpu_ms / r.total_ms << "," << r.max_error << "\n";
  }
  return file.good();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) return EXIT_FAILURE;
  if (options.show_help) {
    PrintUsage();
    return EXIT_SUCCESS;
  }

  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  std::printf("GPU: %s (%d multiprocessors, %.1f GB)\n", properties.name,
              properties.multiProcessorCount,
              properties.totalGlobalMem / 1.0e9);
  std::printf("Batch: %d images of %dx%d, Gaussian radius %d, %d streams\n",
              options.num_images, options.width, options.height,
              options.radius, options.num_streams);

  const size_t pixels = static_cast<size_t>(options.width) * options.height;
  const size_t total_pixels = pixels * options.num_images;

  // Pinned (page-locked) host memory is required for truly asynchronous
  // copies, and is also used for the other GPU versions for a fair comparison.
  float* host_in = nullptr;
  float* host_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&host_in, total_pixels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&host_out, total_pixels * sizeof(float)));
  std::vector<float> reference(total_pixels);

  std::printf("Generating %d synthetic images...\n", options.num_images);
  for (int i = 0; i < options.num_images; ++i) {
    GenerateImage(host_in + i * pixels, options.width, options.height, i);
  }

  const std::vector<float> filter = MakeGaussianFilter(options.radius);
  CUDA_CHECK(cudaMemcpyToSymbol(kFilter, filter.data(),
                                filter.size() * sizeof(float)));

  std::vector<Result> results;
  std::printf("Running CPU reference...\n");
  results.push_back(RunCpu(options, filter, host_in, reference.data()));
  std::printf("Running CUDA global-memory kernel...\n");
  results.push_back(RunGpuSequential("cuda_global_memory", false, options,
                                     host_in, host_out, reference.data()));
  std::printf("Running CUDA shared-memory kernel...\n");
  results.push_back(RunGpuSequential("cuda_shared_memory", true, options,
                                     host_in, host_out, reference.data()));
  std::printf("Running CUDA shared-memory kernel with streams...\n");
  results.push_back(
      RunGpuStreams(options, host_in, host_out, reference.data()));

  PrintResults(results, options.num_images);

  if (!WriteCsv(options.csv_path, results, options)) {
    std::fprintf(stderr, "Failed to write %s\n", options.csv_path.c_str());
  } else {
    std::printf("\nResults written to %s\n", options.csv_path.c_str());
  }

  if (options.save_images) {
    std::filesystem::create_directories(options.output_dir);
    const std::string input_path = options.output_dir + "/input_image0.pgm";
    const std::string output_path =
        options.output_dir + "/output_blurred_image0.pgm";
    if (WritePgm(input_path, host_in, options.width, options.height) &&
        WritePgm(output_path, host_out, options.width, options.height)) {
      std::printf("Saved %s and %s\n", input_path.c_str(), output_path.c_str());
    } else {
      std::fprintf(stderr, "Failed to save images to %s\n",
                   options.output_dir.c_str());
    }
  }

  CUDA_CHECK(cudaFreeHost(host_in));
  CUDA_CHECK(cudaFreeHost(host_out));
  return EXIT_SUCCESS;
}
