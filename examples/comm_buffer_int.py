import triton
import triton.language as tl
import torch

# Define the fused computation and communication kernel
@triton.jit
def fused_matmul_comm_kernel(
    A, B, C, C_comm,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    pid_start: tl.constexpr,
    pid_end: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0) + pid_start

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pids = num_pid_m * num_pid_n

    if pid >= pid_end:
        return

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute block indices
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers to A, B
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator with float32 zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        a_int8 = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0)
        b_int8 = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0)
        # Cast int8 to float32
        a = a_int8.to(tl.float32)
        b = b_int8.to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Cast accumulator to int32 before storing
    acc_int32 = acc.to(tl.int32)

    # Store result into C
    c_ptrs = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc_int32, mask=c_mask)

    # Simulate inter-GPU communication
    # For this example, we'll just write to C_comm the same way
    c_comm_ptrs = C_comm + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_comm_ptrs, acc_int32, mask=c_mask)

# Define the kernel launch function
def launch_kernel(A, B, C, C_comm, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, num_gpus):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pids = num_pid_m * num_pid_n

    pids_per_gpu = triton.cdiv(num_pids, num_gpus)

    for gpu_id in range(num_gpus):
        pid_start = gpu_id * pids_per_gpu
        pid_end = min(pid_start + pids_per_gpu, num_pids)
        grid = (pid_end - pid_start,)
        fused_matmul_comm_kernel[grid](
            A, B, C, C_comm,
            M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            pid_start=pid_start,
            pid_end=pid_end,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

# Main code
if __name__ == "__main__":
    # Define matrix dimensions
    M = 64  # Number of rows of A and C
    N = 64  # Number of columns of B and C
    K = 64  # Number of columns of A and rows of B

    # Strides for row-major matrices
    stride_am = K
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1

    # Number of GPUs (for simulation purposes)
    num_gpus = 2  # Adjust as needed

    # Allocate input and output tensors on the GPU using int8
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')
    C = torch.zeros((M, N), dtype=torch.int32, device='cuda')  # Accumulator needs to be int32

    # Allocate communication buffer
    C_comm = torch.zeros_like(C)

    # Ensure the tensors are contiguous in memory
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    C_comm = C_comm.contiguous()

    # Call the launch_kernel function to execute the fused kernel
    launch_kernel(
        A, B, C, C_comm,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        num_gpus
    )

    # Synchronize to ensure kernel completion
    torch.cuda.synchronize()

    # Verification
    # Convert int8 tensors to float32 for PyTorch matmul and then convert the result back to int32
    C_ref = torch.matmul(A.float(), B.float()).to(torch.int32)

    print(C)
    print(C_ref)
    
    # Check if the results are equal
    if torch.equal(C, C_ref):
        print("Success: The Triton kernel produces the correct result.")
    else:
        print("Error: The results do not match.")
        # Compare C and C_ref element-wise
        mismatch_tensor = torch.ne(C, C_ref)
        num_mismatches = mismatch_tensor.int().sum().item()
        print(f"Number of mismatched elements: {num_mismatches}")

    # Optionally, verify C_comm
    if torch.equal(C_comm, C):
        print("Success: The communication buffer matches the computed result.")
    else:
        print("Note: The communication buffer differs from the computed result.")
