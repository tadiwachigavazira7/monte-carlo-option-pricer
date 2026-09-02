import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path("../data/benchmark.csv")
PLOTS_DIR = Path("../plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load benchmark data
# ============================================================

df = pd.read_csv(DATA_FILE)

# Ensure consistent ordering
df = df.sort_values(["num_paths", "threads"])

# ============================================================
# 1. Runtime vs Number of Threads
# ============================================================

plt.figure(figsize=(10, 6))

for paths in sorted(df["num_paths"].unique()):
    subset = df[df["num_paths"] == paths]

    plt.plot(
        subset["threads"],
        subset["runtime_ms"],
        marker="o",
        label=f"{paths:,} paths"
    )

plt.xlabel("Number of Threads")
plt.ylabel("Runtime (ms)")
plt.title("Monte Carlo Runtime vs Number of Threads")
plt.xticks(sorted(df["threads"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "runtime_vs_threads.png",
    dpi=300
)

plt.close()

# ============================================================
# 2. Speedup vs Number of Threads
# ============================================================

plt.figure(figsize=(10, 6))

for paths in sorted(df["num_paths"].unique()):
    subset = df[df["num_paths"] == paths]

    plt.plot(
        subset["threads"],
        subset["speedup"],
        marker="o",
        label=f"{paths:,} paths"
    )

# Ideal linear speedup
max_threads = df["threads"].max()

plt.plot(
    [1, max_threads],
    [1, max_threads],
    linestyle="--",
    label="Ideal linear speedup"
)

plt.xlabel("Number of Threads")
plt.ylabel("Speedup (×)")
plt.title("Monte Carlo Parallel Speedup")
plt.xticks(sorted(df["threads"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "speedup_vs_threads.png",
    dpi=300
)

plt.close()

# ============================================================
# 3. Parallel Efficiency vs Number of Threads
# ============================================================

plt.figure(figsize=(10, 6))

for paths in sorted(df["num_paths"].unique()):
    subset = df[df["num_paths"] == paths]

    plt.plot(
        subset["threads"],
        subset["efficiency_pct"],
        marker="o",
        label=f"{paths:,} paths"
    )

plt.axhline(
    100,
    linestyle="--",
    label="Ideal efficiency"
)

plt.xlabel("Number of Threads")
plt.ylabel("Parallel Efficiency (%)")
plt.title("Monte Carlo Parallel Efficiency")
plt.xticks(sorted(df["threads"].unique()))
plt.ylim(0, 110)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "efficiency_vs_threads.png",
    dpi=300
)

plt.close()

# ============================================================
# 4. Runtime vs Number of Paths
# ============================================================

plt.figure(figsize=(10, 6))

# Focus on the larger workloads where scaling behavior
# is more meaningful.
for threads in sorted(df["threads"].unique()):
    subset = df[df["threads"] == threads]

    plt.plot(
        subset["num_paths"],
        subset["runtime_ms"],
        marker="o",
        label=f"{threads} threads"
    )

plt.xlabel("Number of Monte Carlo Paths")
plt.ylabel("Runtime (ms)")
plt.title("Runtime vs Monte Carlo Workload")
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "runtime_vs_paths.png",
    dpi=300
)

plt.close()

# ============================================================
# Done
# ============================================================

print("Benchmark plots generated:")
print(f"  {PLOTS_DIR / 'runtime_vs_threads.png'}")
print(f"  {PLOTS_DIR / 'speedup_vs_threads.png'}")
print(f"  {PLOTS_DIR / 'efficiency_vs_threads.png'}")
print(f"  {PLOTS_DIR / 'runtime_vs_paths.png'}")