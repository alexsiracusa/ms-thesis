import numpy as np


def calculate(W, v, n):
    M = np.empty_like(W, dtype=object)
    M.fill(None)

    for _ in range(n):
        for col in range(len(v)):
            mask = W[:, col] != 0
            M[mask, col] = v[col]

        print(M)

        v_temp = v.copy()
        for row in range(1, len(v)):
            mask = M[row] != None
            v[row] = np.max(M[row, mask] + W[row, mask]) if sum(mask) != 0 else None
            # v[row] = np.average(M[row, mask] + W[row, mask]) if sum(mask) != 0 else None
            # v[row] = np.sum(M[row, mask] + W[row, mask]) if sum(mask) != 0 else None
            # v[row] = np.average(v[mask]) + np.sum(W[row, mask]) if sum(mask) != 0 else v[row]
            # v[row] = np.sum(M[row, mask] * W[row, mask] + W[row, mask]) if sum(mask) != 0 else None
            # v[row] = float(np.sum(v_temp[mask]) + np.sum(W[row, mask])) if sum(mask) != 0 else None

        print(v)
        print()


if __name__ == '__main__':
    W = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    W = np.array([
        [0.1875, 0.1875, 0.1875, 0.1875],
        [0.1875, 0.1875, 0.1875, 0.1875],
        [0.1875, 0.1875, 0.1875, 0.1875],
        [0.1875, 0.1875, 0.1875, 0.1875],
    ])

    v = np.array([0, None, None, None], dtype=object)

    n = 4

    calculate(W, v, n)

















