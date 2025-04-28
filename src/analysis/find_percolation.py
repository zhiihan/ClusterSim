import numpy as np
import pickle
import matplotlib.pyplot as plt

shape = [20, 20, 200]
samples = 2
p_vec = np.linspace(0.0, 0.35, 50)
plot_data = np.empty((len(p_vec), 3))

cpu_cores = 2


def has_false_value(lst):
    return any(not item for item in lst)


for pindex, p in enumerate(p_vec):
    s = []
    c = []
    for seed in range(samples):
        with open(f"./data/percol{pindex}shape{shape[2]}sample{seed}s", "rb") as f:
            s.append(pickle.load(f)[0])
        with open(f"./data/percol{pindex}shape{shape[2]}sample{seed}c", "rb") as f:
            c.append(pickle.load(f)[0])
    print(p, s, c)

    plot_data[pindex, 0] = p
    plot_data[pindex, 1] = has_false_value(s)
    plot_data[pindex, 2] = has_false_value(c)

plt.title("percolation")
plt.xlabel("p, probability of losing a node")
plt.ylabel(f"Percolates, shape{shape}")

plt.plot(plot_data[:, 0], plot_data[:, 1], label=f"surfacecode")
plt.plot(plot_data[:, 0], plot_data[:, 2], label=f"cube")
plt.legend()
plt.savefig(f"percolnew.png")
