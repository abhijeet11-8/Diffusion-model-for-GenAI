import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import beta

thetas = [4]#2.5, 3, 3.0001, 3.001, 3.01, 3.2, 3.25, 3.5, 3.6, 3.75, 3.9, 4]
t_total = 1000
transient = 0
x0 = np.linspace(0, 1, t_total)
bins = int(np.round(np.sqrt(t_total)))
smooth_width = 2  # smoothing window for nicer density curves

fig, axes = plt.subplots(4, 4, figsize=(12, len(thetas)))
axes = axes.flat

for i, theta in enumerate(thetas):
    x = x0
    traj = np.empty(t_total)
    for k in range(t_total):
        x = theta * x * (1 - x)
        # traj[k] = x

    data = x#traj  # discard transient
    counts, edges = np.histogram(data, bins=bins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # smooth histogram
    window = np.ones(smooth_width) / smooth_width
    smooth_counts = np.convolve(counts, window, mode='same')

    ax = axes[i]
    ax.plot(centers, smooth_counts, lw=1.5)
    ax.fill_between(centers, smooth_counts, alpha=0.2)
    ax.set_title(f"θ = {theta}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_ylabel("density")

    # Plot 2: Beta(1/5, 1/5) distribution
    alpha_beta = 1/2
    beta_param = 1/2
    x_vals = np.linspace(0, 1, t_total)
    pdf_vals = beta.pdf(x_vals, alpha_beta, beta_param)

    axes[i+1].plot(x_vals, pdf_vals, lw=1.5, label=f'Beta({alpha_beta:.2f}, {beta_param:.2f})')
    axes[i+1].fill_between(x_vals, pdf_vals, alpha=0.2)
    axes[i+1].set_title(f"Beta Distribution: α = {alpha_beta:.2f}, β = {beta_param:.2f}")
    axes[i+1].set_xlim(0, 1)
    axes[i+1].set_ylim(0, None)
    axes[i+1].set_xlabel("x")
    axes[i+1].set_ylabel("density")
    axes[i+1].grid(True, alpha=0.3)
    axes[i+1].legend()

# hide unused subplots (we have 9 slots, 7 used)
for j in range(len(thetas)+1, len(axes)):
    axes[j].axis("off")

plt.savefig("")
plt.tight_layout()
plt.show()