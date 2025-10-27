#!/usr/bin/env python3
"""
GPU Requirements Check Script
Run this script to verify your GPU meets the requirements for the experiment
"""

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

def check_gpu_capabilities():
    """Check if GPU meets the requirements"""
    print("🔍 Checking GPU Capabilities...")
    print("=" * 50)

    # Get GPU device information
    device = cuda.Device(0)

    print(f"GPU Name: {device.name()}")
    print(f"Compute Capability: {device.compute_capability()}")
    print(f"Total Memory: {device.total_memory() / (1024**3):.1f} GB")
    print(f"Multiprocessors: {device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)}")
    print(f"Max Threads Per Block: {device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)}")
    print(f"Max Block Dimensions: {device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X)} x {device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Y)} x {device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Z)}")
    print(f"Max Grid Dimensions: {device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)} x {device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)} x {device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Z)}")

    # Check requirements
    requirements_met = True

    # Check compute capability (need at least 3.5 for good performance)
    compute_capability = device.compute_capability()
    if compute_capability < (3, 5):
        print("❌ Compute capability < 3.5 - Performance may be suboptimal")
        requirements_met = False

    # Check memory
    min_memory = 2 * 1024**3  # 2 GB minimum
    if device.total_memory() < min_memory:
        print(f"❌ Insufficient GPU memory: {device.total_memory()/(1024**3):.1f} GB < 2 GB")
        requirements_met = False
    else:
        print(f"✅ Sufficient GPU memory: {device.total_memory()/(1024**3):.1f} GB")

    # Check CUDA driver version
    driver_version = cuda.get_driver_version()
    print(f"CUDA Driver Version: {driver_version}")

    # Simple kernel compilation test
    try:
        test_kernel = """
        __global__ void test_kernel(int *result) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx == 0) {
                *result = 42;
            }
        }
        """
        mod = SourceModule(test_kernel)
        test_kernel_func = mod.get_function("test_kernel")

        result_gpu = cuda.mem_alloc(4)
        test_kernel_func(result_gpu, block=(1, 1, 1), grid=(1, 1))

        result_host = np.zeros(1, dtype=np.int32)
        cuda.memcpy_dtoh(result_host, result_gpu)

        if result_host[0] == 42:
            print("✅ Kernel compilation and execution test passed")
        else:
            print("❌ Kernel execution test failed")
            requirements_met = False

        result_gpu.free()

    except Exception as e:
        print(f"❌ Kernel test failed: {e}")
        requirements_met = False

    print("=" * 50)
    if requirements_met:
        print("🎉 GPU meets all requirements for the experiment!")
        return True
    else:
        print("⚠️  GPU may have compatibility issues")
        return False

def estimate_max_trap_size():
    """Estimate maximum trap size based on available GPU memory"""
    device = cuda.Device(0)
    total_memory = device.total_memory()

    # Memory requirements per trap entry: 16 bytes (fp + privkey)
    # Bloom filter: ~1.44 * trap_size * -log2(false_positive_rate) bits
    # We'll reserve 50% of memory for trap table and 50% for other operations

    available_memory = total_memory * 0.5  # Use 50% of total memory

    # Memory per trap entry (conservative estimate)
    memory_per_entry = 20  # bytes (16 for data + 4 for overhead)

    max_trap_size = int(available_memory / memory_per_entry)

    print(f"📊 Memory-based estimates:")
    print(f"   Total GPU Memory: {total_memory / (1024**3):.1f} GB")
    print(f"   Available for Trap Table: {available_memory / (1024**3):.1f} GB")
    print(f"   Maximum Recommended Trap Size: {max_trap_size:,} entries")
    print(f"   Memory Usage: {max_trap_size * memory_per_entry / (1024**3):.2f} GB")

    return min(max_trap_size, 2**30)  # Cap at 1 billion entries

if __name__ == "__main__":
    print("GPU Requirements Check for Meet-in-the-Middle Experiment")
    print("=" * 60)

    if check_gpu_capabilities():
        estimate_max_trap_size()
        print("\n✅ Your system is ready for GPU-only experiments!")
    else:
        print("\n❌ Your GPU may not be suitable for this experiment.")
        print("   Consider using a different GPU or cloud instance with better CUDA support.")
