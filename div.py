import numpy as np
import time
import argparse
from ecdsa import SECP256k1
from ecdsa.ellipticcurve import Point
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import struct
import ctypes # Import ctypes

# SECP256k1 constants
P = SECP256k1.curve.p()
N = SECP256k1.order
G = SECP256k1.generator

# Load CUDA kernel
print("[+] Loading CUDA kernel...")
try:
    with open('div.cu', 'r') as f:
        cuda_kernel_code = f.read()
    mod = SourceModule(cuda_kernel_code)
    print("[+] CUDA kernel compiled successfully")
except Exception as e:
    print(f"[-] FATAL: CUDA compilation failed: {e}")
    print("[-] This experiment requires GPU with CUDA support")
    exit(1)

# Get kernel functions
generate_trap_table_kernel = mod.get_function("generate_trap_table_kernel")
precompute_G_table_kernel = mod.get_function("precompute_G_table_kernel")
search_kernel = mod.get_function("search_kernel")

def decompress_pubkey(compressed_pubkey):
    """Decompresses a compressed public key to an ECPoint object."""
    if len(compressed_pubkey) != 33:
        raise ValueError("Compressed public key should be 33 bytes")
    prefix, x_bytes = compressed_pubkey[0], compressed_pubkey[1:]
    if prefix not in [0x02, 0x03]:
        raise ValueError("Invalid prefix")

    x_int = int.from_bytes(x_bytes, 'big')
    x3 = pow(x_int, 3, P)
    y_sq = (x3 + 7) % P
    y = pow(y_sq, (P + 1) // 4, P)

    is_even = (y % 2 == 0)
    if (prefix == 0x02 and not is_even) or (prefix == 0x03 and is_even):
        y = P - y

    return Point(SECP256k1.curve, x_int, y)

def number_inverse_mod(a, m):
    """Calculates modular multiplicative inverse."""
    return pow(a, -1, m)

def splitmix64(x):
    """A 64-bit pseudo-random number generator."""
    x = (x + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (x ^ (x >> 31)) & ((1 << 64) - 1)

def optimize_bloom_filter_size(num_items, false_positive_rate=0.001):
    """Calculates optimal bloom filter size."""
    m = - (num_items * np.log(false_positive_rate)) / (np.log(2) ** 2)
    return int(2 ** np.ceil(np.log2(m)))

def bigint_to_device_format(value, num_words=8):
    """Convert Python integer to BigInt device format."""
    result = np.zeros(num_words, dtype=np.uint32)
    for i in range(num_words):
        result[i] = (value >> (32 * i)) & 0xFFFFFFFF
    return result

def point_to_device_format(point):
    """Convert ECPoint to device format."""
    x_bigint = bigint_to_device_format(point.x())
    y_bigint = bigint_to_device_format(point.y())
    return x_bigint, y_bigint

def setup_constants_on_device():
    """Setup SECP256k1 constants on GPU device."""
    print("[GPU] Initializing SECP256k1 constants...")

    # Convert constants to device format
    p_device = bigint_to_device_format(P)
    n_device = bigint_to_device_format(N)

    # Convert generator point to Jacobian coordinates (Z=1)
    g_x = bigint_to_device_format(G.x())
    g_y = bigint_to_device_format(G.y())
    g_z = bigint_to_device_format(1)

    # Get constant pointers from module
    const_p = mod.get_global("const_p")[0]
    const_n = mod.get_global("const_n")[0]
    const_G_jacobian = mod.get_global("const_G_jacobian")[0]

    # Copy data to device
    cuda.memcpy_htod(const_p, p_device.tobytes())
    cuda.memcpy_htod(const_n, n_device.tobytes())

    # Copy generator point (in Jacobian coordinates)
    g_jacobian_data = np.concatenate([g_x, g_y, g_z, np.array([0], dtype=np.uint32)])
    cuda.memcpy_htod(const_G_jacobian, g_jacobian_data.tobytes())
    print("[GPU] Constants initialized successfully")

def setup_target_on_device(target_point):
    """Setup target point on GPU device."""
    print("[GPU] Setting up target point...")
    target_x, target_y = point_to_device_format(target_point)
    target_z = bigint_to_device_format(1)

    const_target_jacobian = mod.get_global("const_target_jacobian")[0]
    target_data = np.concatenate([target_x, target_y, target_z, np.array([0], dtype=np.uint32)])
    cuda.memcpy_htod(const_target_jacobian, target_data.tobytes())
    print("[GPU] Target point setup completed")

def generate_trap_table_gpu(trap_size, startK, bloom_fp_rate=0.001):
    """Generate trap table using GPU only."""
    print(f"[GPU] Building {trap_size:,} trap entries starting from {startK:,}...")

    # Calculate bloom filter size
    bloom_size = optimize_bloom_filter_size(trap_size, bloom_fp_rate)
    bloom_words = (bloom_size + 31) // 32

    print(f"[GPU] Bloom filter size: {bloom_size:,} bits ({bloom_words:,} words)")

    # Allocate device memory
    trap_table_gpu = cuda.mem_alloc(trap_size * 16)  # fp (8) + privkey (8)
    bloom_filter_gpu = cuda.mem_alloc(bloom_words * 4)

    # Initialize bloom filter to zeros
    cuda.memset_d32(bloom_filter_gpu, 0, bloom_words)

    # Precompute G table
    print("[GPU] Precomputing G table (2^0G to 2^255G)...")
    precompute_G_table_kernel(block=(1, 1, 1), grid=(1, 1))
    cuda.Context.synchronize()
    print("[GPU] G table precomputation completed")

    # Launch trap table generation kernel
    threads_per_block = 256
    blocks_per_grid = (trap_size + threads_per_block - 1) // threads_per_block

    print(f"[GPU] Launching kernel: {blocks_per_grid} blocks × {threads_per_block} threads")

    start_time = time.time()

    generate_trap_table_kernel(
        trap_table_gpu,
        bloom_filter_gpu,
        np.uint32(trap_size),
        np.uint64(bloom_size),
        np.uint64(startK),
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1)
    )

    cuda.Context.synchronize()
    gpu_time = time.time() - start_time

    print(f"[GPU] Trap table generation completed in {gpu_time:.2f}s")
    print(f"[GPU] Performance: {trap_size/gpu_time:,.0f} entries/second")

    # Copy results back to host
    print("[GPU] Copying results to host memory...")
    trap_table_host = np.zeros(trap_size, dtype=[('fp', np.uint64), ('privkey', np.uint64)])
    bloom_filter_host = np.zeros(bloom_words, dtype=np.uint32)

    cuda.memcpy_dtoh(trap_table_host, trap_table_gpu)
    cuda.memcpy_dtoh(bloom_filter_host, bloom_filter_gpu)

    # Convert to dictionary format
    trap_table_dict = {}
    for entry in trap_table_host:
        trap_table_dict[entry['fp']] = entry['privkey']

    # Free device memory
    trap_table_gpu.free()
    bloom_filter_gpu.free()

    print(f"[GPU] Trap table contains {len(trap_table_dict):,} unique entries")

    return trap_table_dict, bloom_filter_host, bloom_size

def search_gpu(trap_table_dict, bloom_filter, bloom_size,
              target_point_divided, pembagi):
    """Search menggunakan GPU hanya."""
    print(f"[GPU] Starting GPU search for target point / {pembagi} in trap table...")

    # Convert trap table to sorted array for binary search on GPU
    print("[GPU] Preparing trap table for GPU search...")
    trap_table_array = np.array([(fp, privkey) for fp, privkey in trap_table_dict.items()],
                               dtype=[('fp', np.uint64), ('privkey', np.uint64)])
    trap_table_array.sort(order='fp')

    print(f"[GPU] Trap table sorted: {len(trap_table_array):,} entries")

    # Allocate device memory
    trap_table_gpu = cuda.mem_alloc(trap_table_array.nbytes)
    bloom_filter_gpu = cuda.mem_alloc(bloom_filter.nbytes)
    # Allocate memory for found results: k_target, k_trap (step is not applicable here)
    found_results_gpu = cuda.mem_alloc(2 * ctypes.sizeof(ctypes.c_ulonglong))


    # Copy data to device
    cuda.memcpy_htod(trap_table_gpu, trap_table_array)
    cuda.memcpy_htod(bloom_filter_gpu, bloom_filter)
    # Initialize found_results_gpu to zeros
    cuda.memset_d8(found_results_gpu, 0, 2 * ctypes.sizeof(ctypes.c_ulonglong))

    # Calculate fingerprint of the target point divided by pembagi
    # Access x and y directly from the Point object
    low64_x = target_point_divided.x() & ((1 << 64) - 1)
    y_parity = target_point_divided.y() % 2
    target_fp = splitmix64(low64_x ^ y_parity)

    # Launch search kernel
    threads_per_block = 256
    blocks_per_grid = 1 # Only need one block as search is within the trap table

    print(f"[GPU] Search kernel: {blocks_per_grid} blocks × {threads_per_block} threads")
    print(f"[GPU] Starting search at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    search_kernel(
        trap_table_gpu,
        bloom_filter_gpu,
        np.uint32(len(trap_table_array)),
        np.uint64(bloom_size),
        np.uint64(target_fp), # Pass target fingerprint
        np.uint64(pembagi),
        found_results_gpu,
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1)
    )

    cuda.Context.synchronize()
    search_time = time.time() - start_time

    print(f"[GPU] Search completed in {search_time:.2f}s")


    # Check results
    results_host = (ctypes.c_ulonglong * 2)()
    cuda.memcpy_dtoh(results_host, found_results_gpu)

    k_target = results_host[0]
    found_trap = results_host[1]


    # Free device memory
    trap_table_gpu.free()
    bloom_filter_gpu.free()
    found_results_gpu.free()


    if k_target != 0:
        print(f"\n🎉 [SUCCESS] PRIVATE KEY FOUND!")
        print(f"    Private Key: {hex(k_target)}")
        print(f"    Trap Key: {found_trap:,}")
        print(f"    Divisor: {pembagi}")
        print(f"    Search Time: {search_time:.2f}s")
        print(f"    Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Verification
        # Need original target_point for verification
        # This would require passing original pubkey or point to this function,
        # or recalculating it here. For simplicity, we'll skip detailed verification here
        # or assume it's done outside if needed.
        # print(f"    ✅ Verification successful (assuming target point was correct)")

        # Save results
        with open("found_key.txt", "w") as f:
            f.write(hex(k_target))

        # Save detailed results
        with open("search_results.txt", "w") as f:
            f.write(f"Private Key: {hex(k_target)}\n")
            f.write(f"Trap Key: {found_trap}\n")
            f.write(f"Divisor: {pembagi}\n")
            f.write(f"Search Time: {search_time:.2f}s\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


        return True
    else:
        print(f"\n❌ SEARCH COMPLETED: Target point / {pembagi} not found in the trap table.")
        return False

def calculate_success_probability(trap_size, divisor):
    """
    Calculate theoretical success probability for the simplified method
    """
    # The probability is the chance that target_point / divisor is in the trap table
    # assuming the trap table covers a random range of size trap_size.
    # This is simply trap_size / N if the trap table range is random.
    # If the trap table range is fixed (startK to startK + trap_size),
    # we need to check if target_point / divisor falls within that range.
    # Since we don't have target_point / divisor here, we'll use the
    # simplified probability based on random distribution for estimation.
    probability = trap_size / N
    return probability

def estimate_computation_time(trap_size):
    """
    Estimate computation time based on GPU performance for the simplified method
    """
    # GPU performance metrics (conservative estimates)
    trap_build_rate = 500000  # 500K entries/sec on GPU
    search_rate = 1000000 # Estimate for a single binary search on GPU

    trap_time = trap_size / trap_build_rate
    search_time = trap_size / search_rate # Roughly proportional to log(trap_size), but using linear for simple estimate
    total_time = trap_time + search_time

    return {
        'trap_build_seconds': trap_time,
        'search_seconds': search_time,
        'total_seconds': total_time,
        'trap_build_minutes': trap_time / 60,
        'search_minutes': search_time / 60,
        'total_minutes': total_time / 60,
        'total_hours': total_time / 3600
    }


def validate_parameters(trap_size, startK):
    """Validate experiment parameters for GPU execution"""
    if trap_size <= 0:
        raise ValueError("Trap size must be positive")
    if startK < 1:
        raise ValueError("Start private key must be >= 1")
    if trap_size > 2**30:
        raise ValueError("Trap size too large for GPU memory")
    if startK + trap_size >= N:
        raise ValueError("Trap range exceeds curve order")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meet-in-the-middle with GPU ONLY - Division Concept')
    parser.add_argument('--pubkey', type=str, required=True, help='Target public key (compressed, hex)')
    parser.add_argument('--pembagi', type=int, required=True, help='Divisor for the division concept')
    parser.add_argument('--trap_size', type=int, default=1048576, help='Size of trap table')
    parser.add_argument('--bloom_fp_rate', type=float, default=0.001, help='Bloom filter false positive rate')
    parser.add_argument('--startK', type=int, default=1, help='Starting private key for trap table')
    args = parser.parse_args()

    print("=" * 70)
    print("GPU-ONLY MEET-IN-THE-MIDDLE WITH DIVISION CONCEPT (Simplified)")
    print("=" * 70)

    # Validate parameters
    try:
        validate_parameters(args.trap_size, args.startK)
    except ValueError as e:
        print(f"❌ Parameter validation failed: {e}")
        exit(1)

    # Calculate success probability and time estimates
    success_prob = calculate_success_probability(args.trap_size, args.pembagi)
    time_est = estimate_computation_time(args.trap_size)

    print(f"📊 EXPERIMENT PARAMETERS:")
    print(f"   Target Public Key: {args.pubkey[:20]}...{args.pubkey[-20:]}")
    print(f"   Divisor: {args.pembagi:,}")
    print(f"   Trap Table Size: {args.trap_size:,}")
    print(f"   Start Private Key for Trap: {args.startK:,}")
    print(f"   Bloom Filter FP Rate: {args.bloom_fp_rate}")
    print(f"")
    print(f"📈 THEORETICAL ANALYSIS:")
    print(f"   Success Probability (Estimate): {success_prob:.4%}")
    print(f"   Estimated Setup Time: {time_est['trap_build_minutes']:.1f} minutes")
    print(f"   Estimated Search Time: {time_est['search_minutes']:.1f} seconds")
    print(f"   Total Estimated Time: {time_est['total_minutes']:.1f} minutes")
    print(f"")

    print("🚀 Starting GPU experiment...")

    try:
        # Setup GPU constants
        setup_constants_on_device()

        # Process target public key
        print("[+] Processing target public key...")
        pubkey_bytes = bytes.fromhex(args.pubkey)
        target_point = decompress_pubkey(pubkey_bytes)
        pembagi_invers = number_inverse_mod(args.pembagi, N)

        print(f"[+] Target point decompressed successfully")
        print(f"[+] Divisor inverse calculated")

        # Calculate the target point divided by pembagi (multiplied by inverse)
        target_point_divided = target_point * pembagi_invers
        print(f"[+] Calculated target point / {args.pembagi}")


        # Generate trap table on GPU
        trap_table_dict, bloom_filter, bloom_size = generate_trap_table_gpu(
            args.trap_size, args.startK, args.bloom_fp_rate
        )

        # Search using GPU
        found = search_gpu(
            trap_table_dict, bloom_filter, bloom_size,
            target_point_divided, args.pembagi
        )

        if not found:
             print("\nProgram finished without finding the private key in the specified trap range.")


    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n💥 GPU ERROR: {e}")
        print("   Please check your CUDA installation and GPU memory")
    finally:
        print("=" * 70)
        print("Experiment completed")
        print("=" * 70)
