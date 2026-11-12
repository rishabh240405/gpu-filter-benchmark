# Project Description

**Replace every [bracketed] item with your own results and experience before
submitting. The "lessons learned" section is graded on whether it shows real
thought, so write about what actually happened to you.**

## Goal
I wanted to understand why GPU programs get fast, not only that they do. I
took one operation (a 2D Gaussian blur on large images) and implemented it
four ways, so that each optimization from the course could be measured
separately: a CPU baseline, a global-memory CUDA kernel, a shared-memory
tiled kernel, and a streams-based version that overlaps data transfer with
computation.

## What the program does
- Generates a batch of [20] synthetic [2048x2048] grayscale images.
- Blurs every image on the CPU (reference) and with three CUDA versions.
- Checks every GPU result against the CPU result (maximum absolute error).
- Reports total time, kernel-only time, per-image time and speedup, and writes
  them to CSV. A command line interface controls batch size, image size,
  filter radius and number of streams.

## Kernels and techniques
1. **Global-memory kernel:** one thread per pixel; each thread reads
   (2r+1)^2 neighbours directly from global memory.
2. **Shared-memory kernel:** each 16x16 block loads its tile plus a halo of r
   pixels into shared memory once; threads then read neighbours from fast
   on-chip memory. Border pixels are handled by clamping coordinates.
3. **Constant memory:** the filter weights are identical for all threads, so
   they live in constant memory.
4. **Streams and pinned memory:** images are distributed over N streams with
   `cudaMemcpyAsync` so one image's copy overlaps with another's kernel.
5. **Timing:** CUDA events for kernel time; wall-clock for end-to-end time.
   A warm-up run keeps one-time driver start-up cost out of the measurements.

## Results
On [GPU name, e.g. Tesla T4] with [20] images of [2048x2048], radius [3]:
- CPU single thread: [___] ms
- Global-memory kernel: [___] ms ([___]x speedup)
- Shared-memory kernel: [___] ms ([___]x speedup)
- Shared memory + [4] streams: [___] ms ([___]x speedup)
- Maximum difference from CPU result: [___] (floating-point rounding only)

## Lessons learned
Write 5-8 sentences in your own words. Ideas, based on what you actually see
in your results:
- [Did shared memory help more at large radius than at small radius? Why?
  Look at your radius sweep (plots/sweep_radius.png).]
- [How much of the total time was copying data vs. running the kernel? What
  did streams change? Look at plots/sweep_streams.png.]
- [A bug or surprise you hit and how you fixed it, e.g. the tile halo, border
  handling, pinned memory requirement, timing the first launch.]
- [What you would try next: separable filters, texture memory, NPP, a real
  image dataset, multiple GPUs.]
