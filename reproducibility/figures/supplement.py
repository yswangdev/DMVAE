"""Supplementary simulation plots and compatibility exports.

The on-disk scenarios are displayed as s01--s04 and compared across six
methods.  The script reads 20 simulation replicates for each method and writes
the two boxplots to ``OUT_DIR``. Posterior-k and real-data supplementary helpers
are re-exported from :mod:`figS1_S2`, matching the manuscript numbering.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figS1_S2 import (
    DEFAULT_PK_MIN,
    DEFAULT_PK_RUN,
    PK_SCENARIOS,
    make_nmi_figures,
    make_plot_extras,
    make_posterior_k,
    make_posterior_k_grid,
    make_sim_nmi_boxplot,
)


# Settings
N_FILES = 20
METHODS = ["scVI", "scGNN", "ADClust", "scACE", "scDAC", "DMVAE"]

# Manuscript scenarios and standardized directory names.
SCENARIOS = ["s01", "s02", "s03", "s04"]
DISPLAY_ORDER = SCENARIOS

# Paths
# Set DMVAE_DIRECTORY to the directory containing Data/ and results/.
DIRECTORY = os.environ.get("DMVAE_DIRECTORY", ".")
SIMULATION_NPZ_ROOT = os.environ.get(
    "SIMULATION_NPZ_ROOT",
    os.path.join(DIRECTORY, "results", "Simulation"),
)
OUT_DIR = os.environ.get(
    "FIGURE_OUTPUT_ROOT",
    os.path.join(DIRECTORY, "results", "dmvae", "figures"),
)

METHOD_FILES = {
    "scVI": "scvi.npz",
    "scGNN": "scgnn.npz",
    "ADClust": "adclust.npz",
    "scACE": "scace.npz",
    "scDAC": "scdac.npz",
    "DMVAE": "dmvae.npz",
}


def _converged(data, best_key: str, trace_key: str) -> float:
    if best_key in data.files:
        return float(np.asarray(data[best_key]).reshape(-1)[-1])
    trace = np.asarray(data[trace_key]).reshape(-1)
    return float(trace[-1]) if trace.size else np.nan


def load_metrics(path: str) -> tuple[float, float]:
    """Load one converged ARI and NMI value from a result archive."""
    try:
        with np.load(path, allow_pickle=True) as data:
            return (
                _converged(data, "BestARI", "ARI"),
                _converged(data, "BestNMI", "NMI"),
            )
    except FileNotFoundError:
        return np.nan, np.nan


def load_dmvae_metrics(path: str) -> tuple[float, float]:
    """Load DMVAE metrics, preferring the explicitly saved best values."""
    return load_metrics(path)


def get_dmvae_path(scenario: str, replicate: int) -> str:
    return os.path.join(
        SIMULATION_NPZ_ROOT,
        scenario,
        f"sim{replicate}",
        METHOD_FILES["DMVAE"],
    )


def collect_results() -> pd.DataFrame:
    """Collect one ARI/NMI record per scenario, replicate, and method."""
    records: list[dict[str, str | float]] = []

    def add(display: str, method: str, path: str) -> None:
        ari, nmi = load_metrics(path)
        records.append(
            {"Scenario": display, "Method": method, "ARI": ari, "NMI": nmi}
        )

    def add_dmvae(display: str, path: str) -> None:
        ari, nmi = load_dmvae_metrics(path)
        records.append(
            {"Scenario": display, "Method": "DMVAE", "ARI": ari, "NMI": nmi}
        )

    for scenario in SCENARIOS:
        for replicate in range(1, N_FILES + 1):
            sim_dir = os.path.join(
                SIMULATION_NPZ_ROOT, scenario, f"sim{replicate}"
            )
            for method in METHODS[:-1]:
                add(
                    scenario,
                    method,
                    os.path.join(sim_dir, METHOD_FILES[method]),
                )

            add_dmvae(scenario, get_dmvae_path(scenario, replicate))

    return pd.DataFrame(records)


PALETTE = ["#8BBDB5", "#508DAB", "#3A528E", "#F39B7F", "#E64B35", "#00A087"]
FLIERPROPS = {
    "marker": "o",
    "markersize": 2,
    "markerfacecolor": "black",
    "markeredgecolor": "black",
    "alpha": 0.8,
}


def make_boxplot(data: pd.DataFrame, metric: str, out_dir: str = OUT_DIR) -> str:
    """Create and save a grouped ARI or NMI boxplot."""
    figure, axis = plt.subplots(figsize=(14, 5), dpi=600)
    sns.boxplot(
        data=data,
        x="Scenario",
        y=metric,
        hue="Method",
        order=DISPLAY_ORDER,
        hue_order=METHODS,
        palette=PALETTE,
        linewidth=0.8,
        flierprops=FLIERPROPS,
        width=0.7,
        gap=0.25,
        ax=axis,
    )
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("")
    axis.set_ylabel(metric, fontsize=20)
    axis.tick_params(axis="x", labelsize=18)
    axis.tick_params(axis="y", labelsize=18)
    axis.legend(
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=len(METHODS),
        frameon=False,
        fontsize=18,
    )
    sns.despine(ax=axis)
    figure.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{metric}_boxplot.png")
    figure.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    data = collect_results()
    make_boxplot(data, "ARI")
    make_boxplot(data, "NMI")


if __name__ == "__main__":
    main()
