import torch
import warnings
import time
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def generate(size, entries, device="cuda"):
    """create a variety of sparse matrices

    Args:
        size (int, optional): Size of the square matrix. Defaults to 1000.
        entries (int, optional): Total number of non-zero entries. Note this is an upper bound, but should be close to the actual size. Defaults to 23*1000.
        device (str, optional): "cuda" or "cpu". Defaults to "cuda".

    Returns:
        dict: Dictionary of the sparse matrices
    """
    output = {}
    indices = torch.randint(0, size, (entries, 2))
    vals = torch.randn(entries)
    coo = torch.sparse_coo_tensor(indices.t(), vals, (size, size), device=device)
    coo = coo.coalesce()

    dense = coo.to_dense()
    csr = coo.to_sparse_csr()

    output["dense"] = dense
    output["csr"] = csr
    return output


## check that all of the matrices are the same operator
def matrix_check(size, entries, device):
    # Generate the matrices
    matrices = generate(size, entries, device=device)
    # Size of the RHS
    x_cols = 100
    x = torch.randn(size, x_cols, device=device)

    print('Matrix-matrix multiplication check')
    y_true = None
    base_name = None

    for matrix, matrix_name in zip(matrices.values(), matrices.keys()):
        y = matrix @ x
        if y_true is None:
            y_true = y
            base_name = matrix_name
        else:
            # print the frobenius norm of the difference
            print(f"||{matrix_name} - {base_name}||_F: {torch.norm(y - y_true)}")




matrix_check(1000, 23*1000, "cuda")


def run_timing(size, entries, device, syncgpu):
    # test if cuda is available
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
            device = "cpu"

    print(f"Running on {device}")
    print(f"Size: {size}")
    print(f"Entries: {entries}")
    print(f"Synchronize GPU: {syncgpu}")

    # Generate the matrices
    matrices = generate(size, entries, device=device)

    # Size of the RHS
    x_cols = 100

    # Compute the timings
    print('Matrix-matrix multiplication timings:')
    base_time = None
    base_name = None
    all_times = {}
    for matrix, matrix_name in zip(matrices.values(), matrices.keys()):
        # first, do a few runs to warm up the cache
        for i in range(2):
            x = torch.randn(size, x_cols, device=device)
            y = matrix @ x

        # now do the timings
        matrix_times = []
        for i in range(5):
            x = torch.randn(size, x_cols, device=device)
            if syncgpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            y = matrix @ x
            if syncgpu:
                torch.cuda.synchronize()
            matrix_time = time.perf_counter() - start
            matrix_times.append(matrix_time)
            print(y.sum().item())
        avg_matrix_time = sum(matrix_times) / len(matrix_times)
        min_matrix_time = min(matrix_times)
        max_matrix_time = max(matrix_times)
        if base_time is None:
            base_time = avg_matrix_time
            base_name = matrix_name

        all_times[matrix_name] = avg_matrix_time

        print(f"{matrix_name} avg time:", avg_matrix_time)
        print(f"{matrix_name} min time:", min_matrix_time)
        print(f"{matrix_name} max time:", max_matrix_time)
        print(f"{matrix_name} speedup over {base_name}:", base_time / avg_matrix_time)

    return all_times


all_times = run_timing(14000, 20000, "cuda", True)

sparsity_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dense_times = []
sparse_times = []

for sparsity in sparsity_values:
    size = 10000
    entries = int(size**2 * sparsity)
    all_times = run_timing(size, entries, "cpu", True)

    dense_times.append(all_times['dense'])
    sparse_times.append(all_times['csr'])

print(dense_times)
print(sparse_times)

plt.figure(figsize=(10, 5))
plt.plot(sparsity_values, dense_times, label='Dense', color='red')
plt.plot(sparsity_values, sparse_times, label='Sparse', color='blue')
# plt.plot(sparsity_values, dense_grad, label='Dense', color='red')
# plt.plot(sparsity_values, sparse_grad, label='Sparse', color='blue')

plt.xlabel('Sparsity')
plt.ylabel('Time')
plt.title('Dot Product Time vs. Sparsity')
plt.legend()
plt.savefig('dot_times.png', dpi=300)
plt.show()