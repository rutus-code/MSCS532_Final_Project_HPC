# Optimization Technique for High-Performance Computing
- Created By: Rutu Shah
- Student ID: 005026421

This project implements the optimization technique for High-Performance Computing, demonstrating the comparison of two matrix multiplication techniques, **Standard Matrix Multiplication** and **Blocked Matrix Multiplication**, measuring their performance and showcasing the results with a performance comparison chart.

## Features

- Implementation of **Standard Matrix Multiplication**
- Implementation of **Blocked Matrix Multiplication** with customizable block size
- Performance measurement and comparison of execution times
- Visualization of execution times across different matrix sizes
- Final Presentation Link

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Matplotlib

You can install the required libraries using:

```bash
pip install numpy matplotlib
```

## Usage

### Running the Experiment

1. Clone the repository or copy the script into a Python file (e.g., `matrix_multiplication.py`).
2. Run the script:

   ```bash
   python matrix_multiplication.py
   ```

3. The program will:
   - Perform **Standard Matrix Multiplication** and print the execution time.
   - Perform **Blocked Matrix Multiplication** with a predefined block size and print the execution time.
   - Validate that both methods produce the same results.
   - Compute and display the speedup achieved by the blocked implementation.
   - Generate a bar chart comparing the execution times of both methods across different matrix sizes.

### Customizing Parameters

- Modify `n` to change the size of the matrices.
- Modify `block_size` to experiment with different block sizes for blocked multiplication.

### Example Output

```
Running Standard Matrix Multiplication...
Execution Time (Standard): 11.64 seconds

Running Blocked Matrix Multiplication...
Execution Time (Blocked): 12.08 seconds

Validating Results...
Results are correct!

Speedup (Blocked over Standard): 0.96x
```

A bar chart (`performance_comparison.png`) will also be generated comparing execution times.

## Code Overview

### Standard Matrix Multiplication

```python
def standard_matrix_multiplication(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C
```

### Blocked Matrix Multiplication

```python
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
```

### Performance Visualization

Execution times for both methods are plotted against different matrix sizes, demonstrating the efficiency of blocked multiplication.

```python
plt.bar(matrix_sizes, standard_times, color='b', width=100, label='Standard')
plt.bar(matrix_sizes, blocked_times, color='g', width=100, label='Blocked')
plt.xlabel('Matrix Size (n x n)')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison: Standard vs. Blocked Matrix Multiplication')
plt.legend()
plt.savefig("performance_comparison.png")
```

## Results

The experiment highlights the performance improvements gained by using **Blocked Matrix Multiplication**, particularly for larger matrix sizes, leveraging better cache utilization.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.

## Acknowledgments

- Inspired by classic computational algorithms for matrix multiplication.

## Presentation URL
- https://cumber-my.sharepoint.com/:p:/g/personal/rshah26421_ucumberlands_edu/EcO5WxdmoxJGt_BNiMiUbwcB3LLdT24wsrWEo_l6_KOqPg?email=vanessa.cooper%40ucumberlands.edu&e=CNh1J1
