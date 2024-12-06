import numpy as np
import time

# Standard Matrix Multiplication
def standard_matrix_multiplication(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Blocked Matrix Multiplication
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

# Function to measure execution time
def measure_execution_time(func, A, B, block_size=None):
    start_time = time.time()
    if block_size:
        result = func(A, B, block_size)
    else:
        result = func(A, B)
    end_time = time.time()
    return end_time - start_time, result

# Main Experiment
def main():
    # Matrix size and block size
    n = 256
    block_size = 64

    # Generate random matrices
    np.random.seed(0)  # For reproducibility
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Standard Matrix Multiplication
    print("Running Standard Matrix Multiplication...")
    standard_time, standard_result = measure_execution_time(standard_matrix_multiplication, A, B)
    print(f"Execution Time (Standard): {standard_time:.4f} seconds")

    # Blocked Matrix Multiplication
    print("Running Blocked Matrix Multiplication...")
    blocked_time, blocked_result = measure_execution_time(blocked_matrix_multiplication, A, B, block_size)
    print(f"Execution Time (Blocked): {blocked_time:.4f} seconds")

    # Validate Results
    print("Validating Results...")
    if np.allclose(standard_result, blocked_result):
        print("Results are correct!")
    else:
        print("Discrepancy in results!")

    # Performance Improvement
    speedup = standard_time / blocked_time
    print(f"Speedup (Blocked over Standard): {speedup:.2f}x")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

# Example data
matrix_sizes = [256, 512, 1024]
standard_times = [11.64, 94.23, 751.18]  
blocked_times = [12.08, 86.45, 670.32]

plt.bar(matrix_sizes, standard_times, color='b', width=100, label='Standard')
plt.bar(matrix_sizes, blocked_times, color='g', width=100, label='Blocked')
plt.xlabel('Matrix Size (n x n)')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison: Standard vs. Blocked Matrix Multiplication')
plt.legend()
plt.show()
plt.savefig("performance_comparison.png")

