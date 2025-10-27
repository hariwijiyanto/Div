import numpy as np
import time
import argparse
from ecdsa import SECP256k1
from ecdsa.ellipticcurve import Point
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import struct
import ctypes

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
    print(f"[GPU] Building {trap_size:,} trap entries...")

    # Calculate bloom filter size
    bloom_size = optimize_bloom_filter_size(trap_size, bloom_fp_rate)
    bloom_words = (bloom_size + 31) // 32

    # Allocate device memory
    trap_table_gpu = cuda.mem_alloc(trap_size * 16)
    bloom_filter_gpu = cuda.mem_alloc(bloom_words * 4)

    # Initialize bloom filter to zeros
    cuda.memset_d32(bloom_filter_gpu, 0, bloom_words)

    # Precompute G table
    precompute_G_table_kernel(block=(1, 1, 1), grid=(1, 1))
    cuda.Context.synchronize()

    # Launch trap table generation kernel
    threads_per_block = 256
    blocks_per_grid = (trap_size + threads_per_block - 1) // threads_per_block

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

    # Copy results back to host
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

    print(f"[GPU] Trap table ready: {len(trap_table_dict):,} entries, {trap_size/gpu_time:,.0f} entries/sec")

    return trap_table_dict, bloom_filter_host, bloom_size

def search_gpu(trap_table_dict, bloom_filter, bloom_size,
              target_point_divided, pembagi, iteration=0, offset=0):
    """Search menggunakan GPU dengan output yang disederhanakan."""
    # Convert trap table to sorted array for binary search on GPU
    trap_table_array = np.array([(fp, privkey) for fp, privkey in trap_table_dict.items()],
                               dtype=[('fp', np.uint64), ('privkey', np.uint64)])
    trap_table_array.sort(order='fp')

    # Allocate device memory
    trap_table_gpu = cuda.mem_alloc(trap_table_array.nbytes)
    bloom_filter_gpu = cuda.mem_alloc(bloom_filter.nbytes)
    found_results_gpu = cuda.mem_alloc(2 * ctypes.sizeof(ctypes.c_ulonglong))

    # Copy data to device
    cuda.memcpy_htod(trap_table_gpu, trap_table_array)
    cuda.memcpy_htod(bloom_filter_gpu, bloom_filter)
    cuda.memset_d8(found_results_gpu, 0, 2 * ctypes.sizeof(ctypes.c_ulonglong))

    # Calculate fingerprint
    low64_x = target_point_divided.x() & ((1 << 64) - 1)
    y_parity = target_point_divided.y() % 2
    target_fp = splitmix64(low64_x ^ y_parity)

    # Launch search kernel
    threads_per_block = 256
    blocks_per_grid = 1

    start_time = time.time()

    search_kernel(
        trap_table_gpu,
        bloom_filter_gpu,
        np.uint32(len(trap_table_array)),
        np.uint64(bloom_size),
        np.uint64(target_fp),
        np.uint64(pembagi),
        found_results_gpu,
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1)
    )

    cuda.Context.synchronize()
    search_time = time.time() - start_time

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
        # Rekonstruksi private key target utama
        original_private_key = (k_target - offset) % N
        if original_private_key == 0:
            original_private_key = N

        print(f"\n🎉 [SUCCESS] Iteration {iteration}")
        print(f"    Offset: {offset:,}G")
        print(f"    Private Key: {hex(original_private_key)}")
        print(f"    Search Time: {search_time:.4f}s")
        print(f"    Speed: {1/search_time:.0f} searches/sec")

        # Verification
        verification_point = original_private_key * G
        if verification_point.x() == target_point.x() and verification_point.y() == target_point.y():
            print(f"    ✅ Verification successful")

        # Save results
        with open("found_key.txt", "w") as f:
            f.write(hex(original_private_key))

        return original_private_key, True, search_time
    else:
        return None, False, search_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meet-in-the-middle with GPU ONLY - Division Concept with Offset Loop')
    parser.add_argument('--pubkey', type=str, required=True, help='Target public key (compressed, hex)')
    parser.add_argument('--pembagi', type=int, required=True, help='Divisor for the division concept')
    parser.add_argument('--trap_size', type=int, default=1000000, help='Size of trap table')
    parser.add_argument('--bloom_fp_rate', type=float, default=0.001, help='Bloom filter false positive rate')
    parser.add_argument('--startK', type=int, default=1, help='Starting private key for trap table')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum number of offset iterations')
    parser.add_argument('--offset_step', type=int, default=1000, help='G point offset step for each iteration')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU MITM WITH DIVISION CONCEPT")
    print("=" * 60)

    # Basic parameter validation
    if args.trap_size <= 0:
        print("❌ Trap size must be positive")
        exit(1)
    if args.startK < 1:
        print("❌ Start private key must be >= 1")
        exit(1)

    print(f"Target: {args.pubkey[:20]}...{args.pubkey[-20:]}")
    print(f"Trap Size: {args.trap_size:,}")
    print(f"Divisor: {args.pembagi:,}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Offset Step: {args.offset_step:,}G")
    print()

    total_start_time = time.time()

    try:
        # Setup GPU constants
        setup_constants_on_device()

        # Process target public key
        pubkey_bytes = bytes.fromhex(args.pubkey)
        target_point = decompress_pubkey(pubkey_bytes)
        pembagi_invers = number_inverse_mod(args.pembagi, N)

        # Generate trap table on GPU (hanya sekali)
        trap_start_time = time.time()
        trap_table_dict, bloom_filter, bloom_size = generate_trap_table_gpu(
            args.trap_size, args.startK, args.bloom_fp_rate
        )
        trap_time = time.time() - trap_start_time

        # Loop dengan offset bertahap
        found = False
        private_key = None
        total_search_time = 0
        successful_iteration = 0
        
        print(f"\nStarting search with {args.max_iterations} iterations...")
        print("Iteration  Status    Offset    Time")
        print("-" * 40)

        for iteration in range(args.max_iterations):
            offset = iteration * args.offset_step
            
            # Hitung target point dengan offset
            if offset == 0:
                current_target_point = target_point
            else:
                offset_point = offset * G
                current_target_point = target_point + offset_point

            # Hitung (target_point + offset*G) / pembagi
            current_target_divided = current_target_point * pembagi_invers

            # Search menggunakan GPU
            private_key, found, search_time = search_gpu(
                trap_table_dict, bloom_filter, bloom_size,
                current_target_divided, args.pembagi, iteration, offset
            )
            
            total_search_time += search_time
            
            if found:
                successful_iteration = iteration
                break
            else:
                # Tampilkan progress setiap 10 iterasi atau iterasi terakhir
                if iteration % 10 == 0 or iteration == args.max_iterations - 1:
                    print(f"{iteration:4d}      {'FAIL':8} {offset:8,}G  {search_time:.4f}s")

        total_time = time.time() - total_start_time

        print("\n" + "=" * 60)
        if found:
            print("🎉 SEARCH SUCCESSFUL!")
            print(f"   Private Key Found: {hex(private_key)}")
            print(f"   Found at iteration: {successful_iteration}")
            print(f"   Final Offset: {successful_iteration * args.offset_step:,}G")
        else:
            print("❌ SEARCH FAILED")
            print(f"   No key found after {args.max_iterations} iterations")
        
        print(f"\n⏱️  PERFORMANCE SUMMARY:")
        print(f"   Trap Table Generation: {trap_time:.2f}s")
        print(f"   Total Search Time: {total_search_time:.2f}s")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average Search Time/Iteration: {total_search_time/args.max_iterations:.4f}s")
        print(f"   Search Speed: {args.max_iterations/total_search_time:.1f} iterations/sec")
        
        if found:
            print(f"   Key Found at Rate: {1/total_time:.6f} keys/sec")
        
        print("=" * 60)

    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
    finally:
        print("Experiment completed")
