import numpy as np

M = np.array([
    [0,   0,   0,   0],
    [1,   0,   0,   0],
    [0,   1,   0,   0],
    [0,   0,   1,   0]
])

n = 6

M = np.eye(n, k=-1)
U = np.ones((3, 3)) - np.eye(3)

def calc(M):
    total = 0.0
    A = np.eye(M.shape[0])
    for k in range (1,n):
        A = A @ M
        total += sum(A[-1]) * k
        # print(A)
        # print()

    # print(A)
    # return total
    return sum(A[-1]) * k

print(f'M: {calc(M)}')
print(f'U: {calc(U) * (n-1) / (n**2)}')







