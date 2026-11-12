#!/usr/bin/env python3
"""Creates charts and a before/after image from the benchmark output.

Usage (from the repository root, after running the benchmark):
    python3 scripts/plot_results.py

Inputs (whichever exist are used):
    output/benchmark_results.csv      main run, from ./run.sh
    results/radius_*.csv              radius sweep, from run_experiments.sh
    results/streams_*.csv             stream-count sweep
    output/input_image0.pgm and output/output_blurred_image0.pgm

Outputs go to the plots/ directory.
"""

import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

PLOT_DIR = "plots"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_main_results(csv_path):
    rows = read_csv(csv_path)
    names = [r["method"] for r in rows]
    total_ms = [float(r["total_ms"]) for r in rows]
    speedup = [float(r["speedup_vs_cpu"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, total_ms, color="#4c72b0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Total time for batch (ms, log scale)")
    axes[0].set_title("Total time")
    axes[1].bar(names, speedup, color="#55a868")
    axes[1].set_ylabel("Speedup vs CPU (x)")
    axes[1].set_title("Speedup")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    for i, value in enumerate(speedup):
        axes[1].text(i, value, f"{value:.1f}x", ha="center", va="bottom")
    first = rows[0]
    fig.suptitle(
        f"{first['num_images']} images of {first['width']}x{first['height']}, "
        f"radius {first['radius']}"
    )
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "main_results.png")
    fig.savefig(out, dpi=150)
    print("Wrote", out)


def plot_sweep(pattern, key_name, title, out_name):
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1)),
    )
    if not files:
        return
    series = {}
    keys = []
    for path in files:
        key = int(re.search(r"_(\d+)\.csv$", path).group(1))
        keys.append(key)
        for row in read_csv(path):
            if row["method"] == "cpu_single_thread":
                continue
            name = re.sub(r"_\d+$", "", row["method"])
            series.setdefault(name, []).append(float(row["speedup_vs_cpu"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, values in series.items():
        ax.plot(keys, values, marker="o", label=name)
    ax.set_xlabel(key_name)
    ax.set_ylabel("Speedup vs CPU (x)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, out_name)
    fig.savefig(out, dpi=150)
    print("Wrote", out)


def make_before_after(crop=512):
    before_path = "output/input_image0.pgm"
    after_path = "output/output_blurred_image0.pgm"
    if not (os.path.exists(before_path) and os.path.exists(after_path)):
        return
    before = Image.open(before_path)
    after = Image.open(after_path)
    width, height = before.size
    size = min(crop, width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    box = (left, top, left + size, top + size)
    canvas = Image.new("L", (size * 2 + 10, size), 255)
    canvas.paste(before.crop(box), (0, 0))
    canvas.paste(after.crop(box), (size + 10, 0))
    out = os.path.join(PLOT_DIR, "before_after_blur.png")
    canvas.save(out)
    print("Wrote", out, "(left: original, right: blurred)")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    main_csv = "output/benchmark_results.csv"
    if os.path.exists(main_csv):
        plot_main_results(main_csv)
    plot_sweep("results/radius_*.csv", "Filter radius",
               "Speedup vs filter radius", "sweep_radius.png")
    plot_sweep("results/streams_*.csv", "Number of streams",
               "Speedup vs number of streams", "sweep_streams.png")
    make_before_after()


if __name__ == "__main__":
    main()
