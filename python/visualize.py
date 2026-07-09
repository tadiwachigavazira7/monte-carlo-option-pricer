"""
Visualizes the output of the C++ Monte Carlo engine:
  - a fan chart of simulated GBM price paths (data/sample_paths.csv)
  - histograms of the terminal call/put payoff distributions (data/payoffs.csv)
  - the single- vs multi-threaded benchmark (data/benchmark.csv), if present

Run the C++ pricer first to generate the CSVs:
    ./build/monte_carlo_pricer --export --out data
    ./build/monte_carlo_pricer --benchmark --out data
then:
    python python/visualize.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def plot_price_paths():
    path = os.path.join(DATA_DIR, "sample_paths.csv")
    paths = np.loadtxt(path, delimiter=",")

    fig, ax = plt.subplots(figsize=(9, 5))
    time_steps = np.linspace(0, 1, paths.shape[1])

    for row in paths:
        ax.plot(time_steps, row, linewidth=0.6, alpha=0.5)

    mean_path = paths.mean(axis=0)
    ax.plot(time_steps, mean_path, color="black", linewidth=2, label="Mean path")

    ax.set_title(f"Simulated GBM Price Paths (n={paths.shape[0]})")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Simulated spot price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "price_paths.png"), dpi=150)
    plt.close(fig)


def plot_payoff_distributions():
    path = os.path.join(DATA_DIR, "payoffs.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(data["call_payoff"], bins=80, color="#3b7dd8", edgecolor="none")
    axes[0].set_title("European Call Payoff Distribution")
    axes[0].set_xlabel("Payoff at expiry")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(data["put_payoff"], bins=80, color="#d84b3b", edgecolor="none")
    axes[1].set_title("European Put Payoff Distribution")
    axes[1].set_xlabel("Payoff at expiry")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "payoff_distributions.png"), dpi=150)
    plt.close(fig)


def plot_benchmark():
    path = os.path.join(DATA_DIR, "benchmark.csv")
    if not os.path.exists(path):
        return

    data = np.genfromtxt(path, delimiter=",", names=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["num_paths"], data["single_ms"], marker="o", label="Single-threaded")
    ax.plot(data["num_paths"], data["multi_ms"], marker="o", label="Multi-threaded")

    ax.set_xscale("log")
    ax.set_title("Monte Carlo Pricing Runtime: Single vs Multi-threaded")
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Runtime (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "benchmark.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_price_paths()
    plot_payoff_distributions()
    plot_benchmark()
    print(f"Saved plots to {os.path.abspath(OUT_DIR)}")
