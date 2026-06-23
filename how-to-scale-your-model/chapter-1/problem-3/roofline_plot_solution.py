import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)


def roofline(B, D, F):
    total_flops = 2 * B * D * F
    flops_time = total_flops / 1.97e14
    comms_time = (2 * B * D + D * F + 2 * B * F) / 8.2e11
    total_time = np.maximum(flops_time, comms_time)
    return total_flops / total_time


roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(9, 5))
plt.plot(bs, roofline_big, label="F = D = 4096", linewidth=2.5)
plt.plot(bs, roofline_small, label="F = D = 1024", linewidth=2.5)

plt.legend(fontsize=11)
plt.xlabel("Batch Size (B)", fontsize=12)
plt.ylabel("Achieved Performance (bfloat16 FLOPs/s)", fontsize=12)
plt.title("Roofline-style Performance Bounds on TPU v5e", fontsize=14, pad=15)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# ============================================================
# SAVE THE PLOT (Recommended)
# ============================================================
plt.savefig("roofline_tpu_v5e.png", dpi=300, bbox_inches="tight")
print("Plot saved as roofline_tpu_v5e.png")

# Optional: also show it (but this can block the shell)
plt.show()
