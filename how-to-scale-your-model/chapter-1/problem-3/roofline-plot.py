import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Performance Bound Model (your formulas)
# ============================================================
BW = 8.2e11  # Memory bandwidth (bytes/sec)
LIMIT = 1.97e14  # Compute roof (FLOPs/sec)


def possible_ops_within_bandwidth(B, X):
    """Your bandwidth-limited performance model."""
    return BW * (2 * B * X**2) / (X**2 + 4 * B * X)


def roofline_calc(B, X):
    """Final performance bound = min(compute roof, bandwidth limit)."""
    return min(LIMIT, possible_ops_within_bandwidth(B, X))


# ============================================================
# Parameters
# ============================================================
bmin, bmax = 1.0, 1000
B_values = np.linspace(bmin, bmax, 500)  # smoother curve

X_values = [1024, 2048, 4096]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # nice color palette

# ============================================================
# Plotting
# ============================================================
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

for X, color in zip(X_values, colors):
    Y_values = [roofline_calc(B, X) for B in B_values]
    ax.plot(B_values, Y_values, label=f"X = {X}", linewidth=2.5, color=color)

# Add horizontal compute roof line for reference
ax.axhline(
    y=LIMIT,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Compute Roof ({LIMIT:.2e} FLOPs/s)",
)

# Styling
ax.set_xlabel("B (Parameter)", fontsize=13)
ax.set_ylabel("Performance (FLOPs/sec)", fontsize=13)
ax.set_title(
    "Performance Bounds Visualization\n(Roofline-style Model)", fontsize=15, pad=15
)
ax.legend(fontsize=11, frameon=True)
ax.grid(True, alpha=0.3)

# Use log scale on X-axis (very common and useful for these plots)
ax.set_xscale("log")
ax.set_xlim(bmin, bmax)

# Format Y-axis nicely
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

plt.tight_layout()
plt.savefig("performance_bounds.png", dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved as performance_bounds.png")
