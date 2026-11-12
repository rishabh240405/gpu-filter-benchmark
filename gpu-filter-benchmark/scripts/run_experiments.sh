#!/bin/bash
# Runs the benchmark for several filter radii and stream counts.
# Results are written to results/*.csv and results/*.txt.
# A smaller batch keeps the CPU reference run reasonably short.
set -e
cd "$(dirname "$0")/.."
make
mkdir -p results

for radius in 1 3 7 15; do
  ./gpu_filter_benchmark.exe --num_images 8 --radius "$radius" \
      --save_images 0 --csv_path "results/radius_${radius}.csv" \
      | tee "results/log_radius_${radius}.txt"
done

for streams in 1 2 4 8; do
  ./gpu_filter_benchmark.exe --num_images 16 --streams "$streams" \
      --save_images 0 --csv_path "results/streams_${streams}.csv" \
      | tee "results/log_streams_${streams}.txt"
done
