import numpy as np
import matplotlib.pyplot as plt


def weighted_path_sum(adj: np.ndarray, max_nodes: int) -> float:
    n = adj.shape[0]
    start, end = 0, n - 1
    total = 0.0

    A_power = np.eye(n)  # A^0
    for k in range(1, max_nodes+1):  # path lengths
        A_power = A_power @ adj  # compute A^k
        total += k * A_power[end, start]

    return total

def total_path(adj: np.ndarray, max_nodes: int) -> float:
    n = adj.shape[0]
    total = 0.0

    A_power = np.eye(n)  # A^0
    for k in range(1, max_nodes+1):  # path lengths
        A_power = A_power @ adj  # compute A^k
        total += np.sum(A_power[-1]) * k

    return total


# Example
# adj = np.array([
#     [0,   0,   0,   0],
#     [1,   0,   0,   0],
#     [0.5, 0,   0,   0],
#     [0,   1, 0.5,   0]
# ])

adj = np.array([
    [0,   0,   0,   0],
    [1,   0,   0,   0],
    [0,   1,   0,   0],
    [0,   0,   1,   0]
])

# adj = np.array([
#     [0,    0,    0,    0   ],
#     [0.25, 0.25, 0.25, 0.25],
#     [0.25, 0.25, 0.25, 0.25],
#     [0.25, 0.25, 0.25, 0.25],
# ])

# adj = np.array([
#     [0,    0,    0,    0   ],
#     [0.33, 0.33, 0.33, 0   ],
#     [0.33, 0.33, 0.33, 0   ],
#     [0.33, 0.33, 0.33, 0   ],
# ])

# adj = np.array([
#     [0.1875, 0.1875, 0.1875, 0.1875],
#     [0.1875, 0.1875, 0.1875, 0.1875],
#     [0.1875, 0.1875, 0.1875, 0.1875],
#     [0.1875, 0.1875, 0.1875, 0.1875],
# ])

print(weighted_path_sum(adj, max_nodes=3))  # -> 2.5


nums = []
for _ in range(10000):
    M = np.random.rand(4, 4)
    M = M / np.sum(M) * 3

    nums.append(total_path(M, max_nodes=3))


plt.hist(np.array(nums), bins=30, edgecolor='black')
plt.savefig("fig.png")
