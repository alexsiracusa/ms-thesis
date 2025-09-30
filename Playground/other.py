import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def structural_complexity(W: np.ndarray, n: int = 3,
                          alpha: float = 1.0,
                          beta: float = 1.0,
                          gamma: float = 1.0):
    """
    Compute structural complexity metrics for a weighted adjacency matrix W.

    Parameters
    ----------
    W : np.ndarray
        Weighted adjacency matrix (square, nonnegative). Entry W[i,j] is weight
        of edge j -> i (i receives from j).
    n : int
        Max path length to consider.
    alpha, beta, gamma : float
        Weights for combining metrics into a single scalar.

    Returns
    -------
    metrics : dict
        Dictionary containing:
            - total_walks : total weighted walks up to length n
            - avg_reachability : average fraction of reachable nodes within n steps
            - f_SCC : fraction of nodes in nontrivial strongly connected components
            - spectral_radius : largest eigenvalue magnitude of W
            - C_struct : composite structural complexity score
    """
    B = W.shape[0]
    assert W.shape[0] == W.shape[1], "Matrix must be square"

    # Boolean adjacency for reachability & SCCs
    A_bool = (W > 0).astype(int)

    # --- Reachability up to n steps ---
    R = np.zeros_like(A_bool)
    P_bool = A_bool.copy()
    for k in range(1, n+1):
        R = R | (P_bool > 0)
        P_bool = (P_bool @ A_bool)  # integer multiplication

    avg_reachability = R.sum() / (B * B)

    # --- Walk counts up to length n (weighted) ---
    total_walks = 0.0
    P = W.copy()
    for k in range(1, n+1):
        total_walks += P.sum()
        P = P @ W

    # --- Spectral radius of weighted adjacency ---
    eigvals = np.linalg.eigvals(W)
    spectral_radius = max(abs(eigvals))

    # --- Strongly connected components (boolean graph) ---
    G = nx.from_numpy_array(A_bool, create_using=nx.DiGraph)
    sccs = list(nx.strongly_connected_components(G))
    nontrivial_nodes = sum(len(scc) for scc in sccs if len(scc) > 1)
    f_SCC = nontrivial_nodes / B

    # --- Composite complexity score ---
    C_struct = (alpha * np.log1p(total_walks) / n
                + beta * f_SCC
                + gamma * avg_reachability)

    return {
        "total_walks": total_walks,
        "avg_reachability": avg_reachability,
        "f_SCC": f_SCC,
        "spectral_radius": spectral_radius,
        "C_struct": C_struct
    }

def matrix_measures(A: np.ndarray):
    # Spectral radius = max(abs(eigenvalues))
    eigvals = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigvals))

    # Frobenius norm
    fro_norm = np.linalg.norm(A, 'fro')

    # Rank
    rank = np.linalg.matrix_rank(A)

    # Singular values
    singular_vals = np.linalg.svd(A, compute_uv=False)
    nuclear_norm = np.sum(singular_vals)   # sum of singular values
    spectral_norm = np.max(singular_vals)  # largest singular value

    # Degree variance (row sums as out-degree)
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    out_degree_var = np.var(row_sums)
    in_degree_var = np.var(col_sums)

    return {
        "spectral_radius": spectral_radius,
        "frobenius_norm": fro_norm,
        "rank": rank,
        "singular_values": singular_vals,
        "nuclear_norm": nuclear_norm,
        "spectral_norm": spectral_norm,
        "out_degree_variance": out_degree_var,
        "in_degree_variance": in_degree_var,
    }


if __name__ == "__main__":
    # Example: your 4x4 block-density matrix
    D = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    # D = np.array([
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.5, 0.0, 0.0, 0.0],
    #     [0.0, 0.5, 0.0, 0.0],
    #     [0.0, 0.0, 0.5, 0.0]
    # ])

    U = np.array([
        [0.1875, 0.1875, 0.1875, 0.1875],
        [0.1875, 0.1875, 0.1875, 0.1875],
        [0.1875, 0.1875, 0.1875, 0.1875],
        [0.1875, 0.1875, 0.1875, 0.1875],
    ])

    # D = np.array([
    #     [0.0, 0.5, 0.0, 0.0],
    #     [0.5, 0.0, 0.5, 0.0],
    #     [0.0, 0.5, 0.0, 0.5],
    #     [0.0, 0.0, 0.5, 0.0]
    # ])

    # D = np.array([
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0]
    # ])

    metrics = structural_complexity(D, n=3)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print()

    metrics = matrix_measures(D)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print()

    metrics = matrix_measures(U)
    for k, v in metrics.items():
        print(f"{k}: {v}")


    # Graph random
    # nums = []
    # for _ in range(10000):
    #     n = 10
    #     M = np.random.rand(n, n)
    #     M = M / np.sum(M) * (n - 1)
    #
    #     metrics = structural_complexity(M, n=n - 1)
    #     nums.append(metrics['total_walks'])
    #
    # print(min(nums), max(nums))
    # plt.hist(np.array(nums), bins=30, edgecolor='black')
    # plt.savefig("fig.png")

