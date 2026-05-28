"""Plot Broca 2.0 training curves from training.log.

Reads the TSV log produced by cmd/cortex-broca-train (Pro version) and
renders a 4-panel PNG: val_loss + train_ema, perplexity, grad_norm,
tokens/sec. The PNG sits next to the log so it can be opened in any
image viewer without spinning up Jupyter.

Usage:
    python scripts/plot_training.py [path/to/training.log] [-o out.png]

Defaults to ./data/cortex-auto/training.log and writes
training-curves.png in the same directory.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path


def load_log(path: Path):
    """Return a dict of column-name -> list[float], skipping the header."""
    cols: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for h in header:
            cols[h] = []
        for row in reader:
            if not row or len(row) != len(header):
                continue
            for h, v in zip(header, row):
                try:
                    cols[h].append(float(v))
                except ValueError:
                    cols[h].append(float("nan"))
    return cols


def plot(cols: dict[str, list[float]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed. Run: pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(2)

    steps = cols.get("step", [])
    if not steps:
        print("empty log — no steps to plot", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Broca 2.0 training — {len(steps)} eval points", fontsize=14)

    # Top-left: losses
    ax = axes[0, 0]
    if "train_loss_ema" in cols:
        ax.plot(steps, cols["train_loss_ema"], label="train (EMA)", color="tab:blue")
    if "val_loss" in cols:
        ax.plot(steps, cols["val_loss"], label="val", color="tab:orange", linewidth=2)
    ax.set_ylabel("cross-entropy / token")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: perplexity (log scale because it spans orders of magnitude)
    ax = axes[0, 1]
    if "val_ppl" in cols:
        ax.semilogy(steps, cols["val_ppl"], color="tab:red")
        ax.set_ylabel("perplexity (log)")
    ax.set_title("Validation perplexity")
    ax.grid(True, alpha=0.3, which="both")

    # Bottom-left: grad norm
    ax = axes[1, 0]
    if "grad_norm" in cols:
        ax.plot(steps, cols["grad_norm"], color="tab:green")
    ax.set_ylabel("L2 norm")
    ax.set_xlabel("step")
    ax.set_title("Gradient norm (pre-clip)")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="clip threshold")
    ax.legend()

    # Bottom-right: throughput
    ax = axes[1, 1]
    if "tokens_per_sec" in cols:
        ax.plot(steps, cols["tokens_per_sec"], color="tab:purple")
    ax.set_ylabel("tokens/sec")
    ax.set_xlabel("step")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def summarise(cols: dict[str, list[float]]) -> str:
    """Short text summary printed to stdout alongside the PNG."""
    if not cols.get("step"):
        return "empty log"
    n = len(cols["step"])
    first_step = int(cols["step"][0])
    last_step = int(cols["step"][-1])
    parts = [f"{n} eval points covering step {first_step} -> {last_step}"]
    if "val_loss" in cols and cols["val_loss"]:
        v0 = cols["val_loss"][0]
        v1 = cols["val_loss"][-1]
        best = min(cols["val_loss"])
        best_idx = cols["val_loss"].index(best)
        best_step = int(cols["step"][best_idx])
        parts.append(
            f"val_loss: {v0:.4f} -> {v1:.4f} (best {best:.4f} @ step {best_step})"
        )
    if "val_ppl" in cols and cols["val_ppl"]:
        parts.append(
            f"val_ppl: {cols['val_ppl'][0]:.1f} -> {cols['val_ppl'][-1]:.1f}"
        )
    if "grad_norm" in cols and cols["grad_norm"]:
        gmax = max(cols["grad_norm"])
        parts.append(f"grad_norm max: {gmax:.2f}")
    if "tokens_per_sec" in cols and cols["tokens_per_sec"]:
        tavg = sum(cols["tokens_per_sec"]) / len(cols["tokens_per_sec"])
        parts.append(f"avg throughput: {tavg:.0f} tok/s")
    return "\n  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        default="./data/cortex-auto/training.log",
        help="Path to training.log (default: ./data/cortex-auto/training.log)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output PNG path (default: training-curves.png next to log)",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"log not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.out) if args.out else log_path.with_name("training-curves.png")

    cols = load_log(log_path)
    plot(cols, out_path)
    print(f"Wrote {out_path}")
    print("  " + summarise(cols))


if __name__ == "__main__":
    main()
