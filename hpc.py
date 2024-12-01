import numpy as np
import time

# Function for standard matrix multiplication (baseline)
def standard_matrix_multiplication(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Function for blocked matrix multiplication (optimized)
def blocked_matrix_multiplication(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n))
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        for k in range(kk, min(kk + block_size, n)):
                            C[i, j] += A[i, k] * B[k, j]
    return C

# Generate random matrices for testing
n = 256  # Matrix size (n x n)
block_size = 64  # Block size for optimization
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# Measure runtime for standard matrix multiplication
start_standard = time.time()
C_standard = standard_matrix_multiplication(A, B)
end_standard = time.time()

# Measure runtime for blocked matrix multiplication
start_blocked = time.time()
C_blocked = blocked_matrix_multiplication(A, B, block_size)
end_blocked = time.time()

# Print results
print(f"Standard Matrix Multiplication Time: {end_standard - start_standard:.4f} seconds")
print(f"Blocked Matrix Multiplication Time: {end_blocked - start_blocked:.4f} seconds")

# Verify the results are similar (small numerical differences due to floating-point arithmetic)
if np.allclose(C_standard, C_blocked):
    print("The results of both methods are consistent!")
else:
    print("The results differ!")

# Save results for analysis
np.savetxt("matrix_standard.csv", C_standard, delimiter=",")
np.savetxt("matrix_blocked.csv", C_blocked, delimiter=",")


