def possible_ops_within_bandwidth(B, X):
    BW = 8.2e11

    return BW * (2 * B * X) / (X**2 + 4 * B * X)

def roofline_calc(B, X):
    LIMIT = 1.97e14

    return min(LIMIT, possible_ops_within_bandwidth(B, X))

bmin = 1.0
bmax = 1000

# plot roofline_calc for B in range(bmin, bmax + 1) and X = [1024, 2048, 4096]
import matplotlib.pyplot as plt

X_values = [1024, 2048, 4096]
B_values = range(int(bmin), int(bmax) + 1)

for X in X_values:
    Y_values = [roofline_calc(B, X) for B in B_values]
    plt.plot(B_values, Y_values, label=f'X = {X}')

plt.xlabel('B')
plt.ylabel('Roofline')
plt.legend()
plt.show()
