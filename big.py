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

def load_cuda_kernel():
    """Load CUDA kernel from file"""
    print("[+] Loading CUDA kernel with big integer support...")
    try:
        with open('div_bigint.cu', 'r') as f:
            cuda_kernel_code = f.read()
        mod = SourceModule(cuda_kernel_code)
        print("[+] CUDA kernel compiled successfully")
        return mod
    except Exception as e:
        print(f"[-] FATAL: CUDA compilation failed: {e}")
        print("[-] This experiment requires GPU with CUDA support")
        exit(1)

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
    """Convert Python big integer to BigInt device format."""
    result = np.zeros(num_words, dtype=np.uint32)
    for i in range(num_words):
        result[i] = (value >> (32 * i)) & 0xFFFFFFFF
    return result

def setup_constants_on_device(mod):
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

def setup_startK_on_device(mod, startK):
    """Setup startK big integer on GPU device."""
    startK_device = bigint_to_device_format(startK)
    const_startK = mod.get_global("const_startK")[0]
    cuda.memcpy_htod(const_startK, startK_device.tobytes())

def generate_trap_table_gpu_bigint(mod, trap_size, startK, bloom_fp_rate=0.001):
    """Generate trap table using GPU dengan big integer support."""
    print(f"[GPU] Building {trap_size:,} trap entries...")

    # Get kernel functions
    generate_trap_table_kernel_bigint = mod.get_function("generate_trap_table_kernel_bigint")
    precompute_G_table_kernel = mod.get_function("precompute_G_table_kernel")

    # Calculate bloom filter size
    bloom_size = optimize_bloom_filter_size(trap_size, bloom_fp_rate)
    bloom_words = (bloom_size + 31) // 32

    # Setup startK di device
    setup_startK_on_device(mod, startK)

    # Allocate device memory
    trap_table_gpu = cuda.mem_alloc(trap_size * 16)  # fp (8) + privkey (8)
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

    generate_trap_table_kernel_bigint(
        trap_table_gpu,
        bloom_filter_gpu,
        np.uint32(trap_size),
        np.uint64(bloom_size),
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

    # Reconstruct full private keys menggunakan startK + index
    trap_table_dict = {}
    for i, entry in enumerate(trap_table_host):
        full_privkey = startK + int(entry['privkey'])  # Konversi ke Python int
        trap_table_dict[entry['fp']] = full_privkey

    # Free device memory
    trap_table_gpu.free()
    bloom_filter_gpu.free()

    print(f"[GPU] Trap table ready: {len(trap_table_dict):,} entries, {trap_size/gpu_time:,.0f} entries/sec")

    return trap_table_dict, bloom_filter_host, bloom_size, startK

def safe_bigint_operation(found_privkey_from_trap, pembagi, offset_val, N_val):
    """Operasi big integer yang aman untuk merekonstruksi kunci asli."""
    try:
        # Gunakan Python big integer untuk semua operasi
        intermediate = found_privkey_from_trap * pembagi
        result = (intermediate - offset_val) % N_val
        return result if result != 0 else N_val
    except OverflowError:
        # print(f"   💥 Overflow in big integer operation") # Suppress verbose debug
        return None

def search_gpu_bigint(mod, trap_table_dict, bloom_filter, bloom_size, startK,
                     target_point_divided, pembagi, original_target_point,
                     iteration=0, offset=0, verbose=False):
    """Search dengan big integer support."""

    # Get kernel function
    search_kernel = mod.get_function("search_kernel")

    # Convert trap table to array untuk GPU
    # Pastikan privkey (index) muat di uint64_t
    trap_table_array = np.array([(fp, int(privkey - startK))
                                for fp, privkey in trap_table_dict.items()],
                               dtype=[('fp', np.uint64), ('privkey', np.uint64)])

    # Hitung fingerprint target
    low64_x = target_point_divided.x() & ((1 << 64) - 1)
    y_parity = target_point_divided.y() % 2
    target_fp = splitmix64(low64_x ^ y_parity)

    # Allocate device memory
    trap_table_gpu = cuda.mem_alloc(trap_table_array.nbytes)
    bloom_filter_gpu = cuda.mem_alloc(bloom_filter.nbytes)
    found_results_gpu = cuda.mem_alloc(2 * ctypes.sizeof(ctypes.c_ulonglong))

    # Copy data to device
    cuda.memcpy_htod(trap_table_gpu, trap_table_array)
    cuda.memcpy_htod(bloom_filter_gpu, bloom_filter)
    cuda.memset_d8(found_results_gpu, 0, 2 * ctypes.sizeof(ctypes.c_ulonglong))

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
        np.uint64(pembagi & 0xFFFFFFFFFFFFFFFF),  # Hanya 64-bit LSB
        found_results_gpu,
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1)
    )

    cuda.Context.synchronize()
    search_time = time.time() - start_time

    # Check results
    results_host = np.zeros(2, dtype=np.uint64)
    cuda.memcpy_dtoh(results_host, found_results_gpu)

    found_index = int(results_host[0])  # Konversi ke Python int
    found_flag = results_host[1]

    # Free device memory
    trap_table_gpu.free()
    bloom_filter_gpu.free()
    found_results_gpu.free()

    if found_flag == 1:
        # Ditemukan di GPU
        # Rekonstruksi kunci privat dari trap table menggunakan startK dan index
        found_privkey_from_trap = startK + found_index

        # Rekonstruksi private key target utama dengan operasi big integer yang aman
        original_private_key = safe_bigint_operation(found_privkey_from_trap, pembagi, offset, N)

        if original_private_key is None:
            # print(f"   ⚠️  Failed to reconstruct original private key for index {found_index}") # Suppress verbose debug
            return None, False, search_time

        # Verification dengan original_target_point
        try:
            verification_point = original_private_key * G
            if verification_point.x() == original_target_point.x() and verification_point.y() == original_target_point.y():
                # Save results only on successful verification
                try:
                    with open("found_key.txt", "w") as f:
                        f.write(hex(original_private_key))
                except Exception as e:
                    print(f"    ⚠️  Error saving result: {e}")
                return original_private_key, True, search_time
            else:
                # Verification failed for this candidate
                # if verbose: # Suppress verbose debug
                #     print(f"    ❌ Verification failed for reconstructed key: {hex(original_private_key)}")
                return None, False, search_time
        except Exception as e:
            print(f"    ⚠️  Verification error: {e}")
            return None, False, search_time

    return None, False, search_time

def validate_parameters_bigint(trap_size, startK, pembagi): # Removed max_iterations, offset_step from validation
    """Validasi parameter untuk big integer support."""

    errors = []

    if trap_size <= 0:
        errors.append("Trap size must be positive")
    if startK < 1:
        errors.append("Start private key must be >= 1")
    if startK >= N:
        errors.append("Start private key must be less than curve order N")
    if pembagi < 1:
        errors.append("Divisor must be >= 1")
    # Note: Checking startK + trap_size >= N is tricky with big ints,
    # rely on the search logic to handle wrap-around implicitly
    # if max_iterations < 1: # Validation moved to main loop logic
    #     errors.append("Max iterations must be >= 1")
    # if offset_step < 1: # Validation moved to main loop logic
    #      errors.append("Offset step must be >= 1")


    if errors:
        print("❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"   - {error}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description='Meet-in-the-middle with BIG INTEGER support (up to 2²⁵⁶-1)')
    parser.add_argument('--pubkey', type=str, required=True, help='Target public key (compressed, hex)')
    parser.add_argument('--pembagi', type=str, required=True, help='Divisor as hex string or decimal')
    parser.add_argument('--startK', type=str, required=True, help='Starting private key for trap table as hex string or decimal')
    parser.add_argument('--trap_size', type=int, default=1000000, help='Size of trap table')
    parser.add_argument('--bloom_fp_rate', type=float, default=0.001, help='Bloom filter false positive rate')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum number of offset iterations')
    parser.add_argument('--offset_step', type=int, default=1000, help='G point offset step for each iteration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU MITM WITH BIG INTEGER SUPPORT")
    print("=" * 60)

    # Parse big integer parameters
    try:
        if args.pembagi.startswith('0x'):
            pembagi = int(args.pembagi, 16)
        else:
            pembagi = int(args.pembagi)

        if args.startK.startswith('0x'):
            startK = int(args.startK, 16)
        else:
            startK = int(args.startK)
    except ValueError as e:
        print(f"❌ Invalid big integer format: {e}")
        exit(1)

    # Validate core parameters
    if not validate_parameters_bigint(
        args.trap_size, startK, pembagi
    ):
        exit(1)

    # Validate iteration parameters separately
    if args.max_iterations < 1:
         print("❌ VALIDATION ERROR: Max iterations must be >= 1")
         exit(1)
    if args.offset_step < 1:
         print("❌ VALIDATION ERROR: Offset step must be >= 1")
         exit(1)


    print(f"Target: {args.pubkey}")
    print(f"Divisor: {hex(pembagi)}")
    print(f"StartK: {hex(startK)}")
    print(f"Trap Size: {args.trap_size:,}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Offset Step: {args.offset_step:,}G")
    if args.verbose:
        print("Verbose output enabled.")
    print()


    total_start_time = time.time()
    original_target_point = None

    try:
        # Load CUDA kernel
        mod = load_cuda_kernel()

        # Setup GPU constants
        setup_constants_on_device(mod)

        # Process target public key
        print("[+] Processing target public key...")
        pubkey_bytes = bytes.fromhex(args.pubkey)
        original_target_point = decompress_pubkey(pubkey_bytes)
        # Calculate inverse only once
        pembagi_invers = number_inverse_mod(pembagi, N)
        print(f"[+] Target point decompressed successfully")
        print(f"[+] Divisor inverse calculated: {hex(pembagi_invers)}")


        # Generate trap table on GPU (hanya sekali)
        trap_start_time = time.time()
        trap_table_dict, bloom_filter, bloom_size, startK_used = generate_trap_table_gpu_bigint(
            mod, args.trap_size, startK, args.bloom_fp_rate
        )
        trap_time = time.time() - trap_start_time
        print(f"\nTrap table generation completed in {trap_time:.2f}s")

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
                current_target_point = original_target_point
            else:
                # Calculate offset point using Python's big integer support
                offset_point = offset * G
                # Add offset point to the original target point
                current_target_point = original_target_point + offset_point

            # Hitung (target_point + offset*G) / pembagi
            # This is equivalent to (TargetPoint + offset*G) * pembagi_invers
            current_target_divided = current_target_point * pembagi_invers

            # Search menggunakan GPU
            private_key, found, search_time = search_gpu_bigint(
                mod, trap_table_dict, bloom_filter, bloom_size, startK_used,
                current_target_divided, pembagi, original_target_point, # Pass original_target_point for final verification
                iteration, offset, args.verbose # Pass iteration, offset, verbose for printing
            )

            total_search_time += search_time

            if found:
                successful_iteration = iteration
                break
            else:
                # Hanya tampilkan progress setiap 10 iterasi atau iterasi terakhir
                if iteration % 10 == 9 or iteration == args.max_iterations - 1 or args.verbose:
                     print(f"{iteration+1:4d}      {'FAIL':8} {offset:8,}G  {search_time:.4f}s")
                     if args.verbose:
                          low64_x_verbose = current_target_divided.x() & ((1 << 64) - 1)
                          y_parity_verbose = current_target_divided.y() % 2
                          target_fp_verbose = splitmix64(low64_x_verbose ^ y_parity_verbose)
                          print(f"[Verbose] Target FP: {target_fp_verbose}")


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
        print(f"   Search Speed: {args.max_iterations/max(1, total_search_time):.1f} iterations/sec")

        if found:
            print(f"   Key Found at Rate: {1/total_time:.6f} keys/sec")

        print("=" * 60)

    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Experiment completed")

if __name__ == '__main__':
    main()
