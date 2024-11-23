import threading
import queue

# def multiply_matrices(A, B, result_queue):
#     """Thread 1: Multiply matrices A and B, sending partial results to Thread 2."""
#     rows_A, cols_A = len(A), len(A[0])
#     rows_B, cols_B = len(B), len(B[0])
#     if cols_A != rows_B:
#         raise ValueError("Cannot multiply matrices: incompatible dimensions.")
#     result = [[0] * cols_B for _ in range(rows_A)]
#     for i in range(rows_A):
#         for j in range(cols_B):
#             result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
#             result_queue.put((i, j, result[i][j]))
#             print(f"Thread 1: Sent partial result ({i}, {j}) = {result[i][j]}")
#     result_queue.put(None)

INT_MAX = 2**63 - 1

def generic_matrix_mult(A, B, out):
    """this is the matrix mult kernel"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("Cannot multiply matrices: incompatible dimensions.")

    x = 0
    for i in range(rows_A):
        while (A[i][cols_A - 1] == INT_MAX):
            x += 1
        for j in range(cols_B):
            out[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))


def launch_kernel(matrices):
    n = len(matrices)
    zero_matrices = [[[INT_MAX for _ in range(len(matrices[0][0]))] for _ in range(len(matrices[0]))] for _ in range(n-1)]
    zero_matrices.insert(0, matrices[0])

    threads = []

    for i in range(1, n):
        t = threading.Thread(target=generic_matrix_mult, args=(zero_matrices[i-1], matrices[i], zero_matrices[i]))
        threads.append(t)
    
    for i in range(len(threads)):
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()

    print(zero_matrices[-1])





# def compute_final_result(partial_result_queue, C, final_result_queue):
#     """Thread 2: Use partial results and matrix C to compute the final result."""
#     rows_C, cols_C = len(C), len(C[0])
#     final_result = [[0] * cols_C for _ in range(len(C))]
#     while True:
#         data = partial_result_queue.get()
#         if data is None:
#             break
#         i, j, partial_value = data
#         for k in range(cols_C):
#             final_result[i][k] += partial_value * C[j][k]
#             print(f"Thread 2: Updated final result ({i}, {k}) = {final_result[i][k]}")
#     final_result_queue.put(final_result)
#     print("Thread 2: Final result computed.")

def main():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = [[9, 10], [11, 12]]
    D = [[13, 14], [15, 16]]

    launch_kernel([A, B, C, D])
    # launch_kernel(matrices)
    # partial_result_queue = queue.Queue()
    # final_result_queue = queue.Queue()
    # thread1 = threading.Thread(target=multiply_matrices, args=(A, B, partial_result_queue))
    # thread2 = threading.Thread(target=compute_final_result, args=(partial_result_queue, C, final_result_queue))
    # thread1.start()
    # thread2.start()
    # thread1.join()
    # thread2.join()
    # final_result = final_result_queue.get()
    # print("Final Result Matrix:")
    # for row in final_result:
    #     print(row)

if __name__ == "__main__":
    main()