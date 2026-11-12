#!/bin/bash
# Builds the project, runs the default benchmark, and saves the log.
# Extra arguments are passed to the program, e.g. ./run.sh --radius 7
set -e
make
mkdir -p output
./gpu_filter_benchmark.exe "$@" | tee output/execution_log.txt
