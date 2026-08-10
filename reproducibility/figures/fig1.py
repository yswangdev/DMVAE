"""Figure 1 -- the two code-generated panels of the DMVAE schematic: the S(K)
bar chart, the illustrative latent embedding space, and the p(K) prior histogram.

    python fig1.py
    python fig1.py --prior uniform --out-dir /path/out
"""

from __future__ import annotations

import argparse
import os
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Ellipse, Rectangle

OUTPUT_DIR = "/Volumes/SSD/MCW/Research/Aim 1/Documents/Paper_draft/papers"

TITLE_Y = 1.03
PRIOR_BAR_COLOR = "#bdd8c8"
SELECTED_PRIOR_BAR_COLOR = "#7fa893"
BAR_EDGE_COLOR = "#666666"

A, B = 2, 10
LAM = 4.0
K_ILLUSTRATION = 4
N_POINTS = 300
SELECTED_K = 4


def compute_prior(prior_type: str, a: int = A, b: int = B, lam: float = LAM):
    """Return (k_vals, p_k) for the chosen prior, normalised over [a, b]."""
    k_vals = np.arange(a, b + 1)
    if prior_type == "uniform":
        p_k = np.ones_like(k_vals, dtype=float)
    else:
        p_k = np.array([np.exp(-lam) * lam**k / factorial(k) for k in k_vals])
    return k_vals, p_k / p_k.sum()


def draw_support_panel(ax, k_vals, p_k, selected_k: int = SELECTED_K):
    colors = [PRIOR_BAR_COLOR] * len(k_vals)
    if selected_k in k_vals:
        colors[list(k_vals).index(selected_k)] = SELECTED_PRIOR_BAR_COLOR

    ax.bar(k_vals, p_k, width=0.7, color=colors,
           edgecolor=BAR_EDGE_COLOR, linewidth=0.8)

    ax.set_xlabel("K", fontsize=20)
    ax.set_ylabel("S(K)", fontsize=20)
    ax.set_xticks(k_vals)
    ax.set_xticklabels(k_vals, fontsize=16)
    ax.set_yticks([])
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(False)


def draw_latent_panel(ax, k_illustration: int = K_ILLUSTRATION, n_points: int = N_POINTS):
    pi_k = np.random.dirichlet(alpha=np.ones(k_illustration))

    min_sep = 2.0
    mus: list[np.ndarray] = []
    while len(mus) < k_illustration:
        candidate = np.random.uniform(low=[-4.0, -4.0], high=[4.0, 4.0])
        if all(np.linalg.norm(candidate - m) >= min_sep for m in mus):
            mus.append(candidate)
    mus = np.array(mus)

    all_pts, ellipse_extents = [], []
    for c in range(k_illustration):
        mu_c = mus[c]
        color = f"C{c % 10}"

        cov_x = np.random.uniform(0.6, 1.4)
        cov_y = np.random.uniform(0.3, 1.0)

        ax.add_patch(Ellipse(xy=mu_c, width=2 * cov_x, height=2 * cov_y,
                             edgecolor=color, facecolor="none", linewidth=2))

        n_c = max(10, int(n_points * pi_k[c]))
        cov = np.array([[cov_x**2, 0.0], [0.0, cov_y**2]])
        pts = np.random.multivariate_normal(mu_c, cov, size=n_c)
        all_pts.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=15, alpha=0.75, color=color, clip_on=True)

        ax.text(mu_c[0], mu_c[1] - (cov_y + 0.4),
                r"$\mu_{{{0}|k=4}},\ \sigma_{{{0}|k=4}}^2$".format(c + 1),
                ha="center", fontsize=20)
        ellipse_extents.append([mu_c[0] - cov_x, mu_c[0] + cov_x,
                                mu_c[1] - (cov_y + 0.6), mu_c[1] + cov_y])

    all_pts = np.vstack(all_pts)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    xmin = min(xmin, min(e[0] for e in ellipse_extents))
    xmax = max(xmax, max(e[1] for e in ellipse_extents))
    ymin = min(ymin, min(e[2] for e in ellipse_extents))
    ymax = max(ymax, max(e[3] for e in ellipse_extents))

    pad = 0.25
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$Z_1$", fontsize=14)
    ax.set_ylabel(r"$Z_2$", fontsize=14)
    ax.set_title("Latent embedding space", fontsize=20, y=TITLE_Y)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor="black", linewidth=1.4))
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(False)


def make_frameplot(out_dir: str, prior_type: str = "poisson") -> None:
    k_vals, p_k = compute_prior(prior_type)

    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.30)

    draw_support_panel(fig.add_subplot(gs[0, 0]), k_vals, p_k)
    draw_latent_panel(fig.add_subplot(gs[0, 1]))

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.93])
    plt.savefig(f"{out_dir}/frameplot.png")
    plt.close(fig)
    print(f"Saved {out_dir}/frameplot.png")


def make_uniform_prior_histogram(out_dir: str, a: int = A, b: int = B) -> None:
    k_vals, p_k = compute_prior("uniform", a, b)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.bar(k_vals, p_k, width=0.7, color=PRIOR_BAR_COLOR,
           edgecolor=BAR_EDGE_COLOR, linewidth=0.8)
    ax.set_xlabel("K", fontsize=30)
    ax.set_ylabel("p(K)", fontsize=30)
    ax.set_xticks(k_vals)
    ax.set_xticklabels(k_vals, fontsize=26)
    ax.set_yticks([])
    ax.set_title("Prior distribution of K", fontsize=30, y=TITLE_Y)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/uniform_prior_histogram.png")
    plt.close(fig)
    print(f"Saved {out_dir}/uniform_prior_histogram.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render the code-generated panels of Figure 1.")
    p.add_argument("--prior", choices=["poisson", "uniform"], default="poisson")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    make_frameplot(args.out_dir, args.prior)
    make_uniform_prior_histogram(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
