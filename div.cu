#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

struct TrapEntry {
    uint64_t fp;
    uint64_t privkey;
};

// SECP256k1 constants - will be initialized from host
__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;
__device__ ECPointJac const_G_table[256];
__constant__ ECPointJac const_target_jacobian; // Still needed for precomputation? No, not directly for this search. Can remove later if not used elsewhere.


// Fungsi utilitas BigInt
__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5;
    int bit_idx = i & 31;
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

__host__ __device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)a->data[i] + b->data[i] + carry;
        res->data[i] = (uint32_t)sum;
        carry = (sum >> 32);
    }
}

__host__ __device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t borrow = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t tmp = (uint64_t)a->data[i] - borrow;
        borrow = tmp < b->data[i] ? 1u : 0u;
        res->data[i] = (uint32_t)(tmp - b->data[i]);
    }
}

__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t prod = (uint64_t)a->data[i] * c + carry;
        result[i] = (uint32_t)prod;
        carry = prod >> 32;
    }
    result[8] = (uint32_t)carry;
}

__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < 9; i++) {
        uint64_t sum = (uint64_t)r[i] + addend[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}

__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t prod[2 * BIGINT_WORDS] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            uint64_t tmp = (uint64_t)prod[i + j] + (uint64_t)a->data[i] * b->data[j] + carry;
            prod[i + j] = (uint32_t)tmp;
            carry = tmp >> 32;
        }
        prod[i + BIGINT_WORDS] += (uint32_t)carry;
    }

    BigInt L, H;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        L.data[i] = prod[i];
        H.data[i] = prod[i + BIGINT_WORDS];
    }

    uint32_t Rext[9] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) Rext[i] = L.data[i];
    Rext[8] = 0;

    uint32_t H977[9] = {0};
    multiply_bigint_by_const(&H, 977, H977);
    add_9word(Rext, H977);

    uint32_t Hshift[9] = {0};
    shift_left_word(&H, Hshift);
    add_9word(Rext, Hshift);

    if (Rext[8]) {
        uint32_t extra[9] = {0};
        BigInt extraBI;
        init_bigint(&extraBI, Rext[8]);
        Rext[8] = 0;

        uint32_t extra977[9] = {0}, extraShift[9] = {0};
        multiply_bigint_by_const(&extraBI, 977, extra977);
        shift_left_word(&extraBI, extraShift);

        for (int i = 0; i < 9; i++) extra[i] = extra977[i];
        add_9word(extra, extraShift);
        add_9word(Rext, extra);
    }

    BigInt R_temp;
    convert_9word_to_bigint(Rext, &R_temp);

    if (Rext[8] || compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    if (compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }

    copy_bigint(res, &R_temp);
}

__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

__device__ __forceinline__ void scalar_mod_n(BigInt *res, const BigInt *a) {
    if (compare_bigint(a, &const_n) >= 0) {
        ptx_u256Sub(res, a, &const_n);
    } else {
        copy_bigint(res, a);
    }
}

__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt sum_ab;
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
         uint64_t word_sum = (uint64_t)a->data[i] + b->data[i] + carry;
         sum_ab.data[i] = (uint32_t)word_sum;
         carry = word_sum >> 32;
    }
    if (carry || compare_bigint(&sum_ab, &const_p) >= 0) {
        ptx_u256Sub(res, &sum_ab, &const_p);
    } else {
        copy_bigint(res, &sum_ab);
    }
}

__device__ void modexp(BigInt *res, const BigInt *base, const BigInt *exp) {
    BigInt result;
    init_bigint(&result, 1);
    BigInt b;
    copy_bigint(&b, base);
    for (int i = 0; i < 256; i++) {
         if (get_bit(exp, i)) {
              mul_mod_device(&result, &result, &b);
         }
         mul_mod_device(&b, &b, &b);
    }
    copy_bigint(res, &result);
}

__device__ void mod_inverse(BigInt *res, const BigInt *a) {
    if (is_zero(a)) {
        init_bigint(res, 0);
        return;
    }
    BigInt p_minus_2, two;
    init_bigint(&two, 2);
    ptx_u256Sub(&p_minus_2, &const_p, &two);
    modexp(res, a, &p_minus_2);
}

__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ __forceinline__ void negate_point_jac(ECPointJac *R, const ECPointJac *P) {
    point_copy_jac(R, P);
    sub_mod_device(&R->Y, &const_p, &P->Y);
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device(&X3, &D2, &twoB);
    sub_mod_device(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    BigInt Z2_cubed, Z1_cubed;
    mul_mod_device(&temp, &Z2Z2, &Q->Z); copy_bigint(&Z2_cubed, &temp);
    mul_mod_device(&temp, &Z1Z1, &P->Z); copy_bigint(&Z1_cubed, &temp);
    mul_mod_device(&S1, &P->Y, &Z2_cubed);
    mul_mod_device(&S2, &Q->Y, &Z1_cubed);

    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
            return;
        } else {
            double_point_jac(R, P);
            return;
        }
    }
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&R_big, &S2, &S1);
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    mul_mod_device(&U1H2, &U1, &H2);
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &S1, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    mul_mod_device(&temp, &P->Z, &Q->Z);
    mul_mod_device(&Z3, &temp, &H);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}

// PRECOMPUTATION FUNCTIONS
__global__ void precompute_G_table_kernel() {
    int idx = threadIdx.x;
    if (idx == 0) {
        ECPointJac current = const_G_jacobian;
        point_copy_jac(&const_G_table[0], &current);

        for (int i = 1; i < 256; i++) {
            double_point_jac(&current, &current);
            point_copy_jac(&const_G_table[i], &current);
        }
    }
}

__device__ void scalar_multiply_jac_precomputed(ECPointJac *result, const BigInt *scalar) {
    point_set_infinity_jac(result);

    for (int i = 0; i < 256; i++) {
        if (get_bit(scalar, i)) {
            add_point_jac(result, result, &const_G_table[i]);
        }
    }
}

// FUNGSI BLOOM FILTER
__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ bool test_bit(const uint32_t* bloom_filter, uint64_t index) {
    uint64_t word_index = index / 32;
    uint32_t bit_index = index % 32;
    return (bloom_filter[word_index] >> bit_index) & 1;
}

__device__ bool check_bloom_filter(const uint32_t* bloom_filter, uint64_t bloom_size, uint64_t fp) {
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);

    for (int j = 0; j < 4; j++) {
        uint64_t index = (h1 + j * h2) % bloom_size;
        if (!test_bit(bloom_filter, index)) {
            return false;
        }
    }
    return true;
}

// KERNEL UNTUK GENERASI TRAP TABLE
extern "C"
__global__ void generate_trap_table_kernel(
    TrapEntry* d_trap_table,
    uint32_t* d_bloom_filter,
    uint32_t trap_size,
    uint64_t bloom_size,
    uint64_t startK
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trap_size) {
        return;
    }

    // Hitung private key untuk thread ini
    uint64_t privkey_val = startK + idx;
    BigInt privkey_bi;
    init_bigint(&privkey_bi, 0);
    for (int j = 0; j < 2; j++) {
        privkey_bi.data[j] = (uint32_t)(privkey_val >> (32 * j));
    }

    // Hitung titik publik: G × privkey
    ECPointJac point_jac;
    scalar_multiply_jac_precomputed(&point_jac, &privkey_bi);

    // Konversi ke affine
    ECPoint point_affine;
    jacobian_to_affine(&point_affine, &point_jac);

    // Hitung fingerprint
    uint64_t low64_x = point_affine.x.data[0] | ((uint64_t)point_affine.x.data[1] << 32);
    uint64_t y_parity = (point_affine.y.data[0] & 1) ? 1 : 0;
    uint64_t fp = splitmix64(low64_x ^ y_parity);

    // Simpan ke trap table
    d_trap_table[idx].fp = fp;
    d_trap_table[idx].privkey = privkey_val;

    // Update bloom filter
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);

    for (int j = 0; j < 4; j++) {
        uint64_t index = (h1 + j * h2) % bloom_size;
        uint64_t word_index = index / 32;
        uint32_t bit_index = index % 32;
        atomicOr(&d_bloom_filter[word_index], (1 << bit_index));
    }
}

// KERNEL UNTUK SEARCH DI GPU (VERSI SEDERHANA TANPA STEPS)
extern "C"
__global__ void search_kernel(
    TrapEntry* d_trap_table,
    uint32_t* d_bloom_filter,
    uint32_t trap_table_size,
    uint64_t bloom_size,
    uint64_t target_fp, // Target fingerprint to search for
    uint64_t pembagi,
    unsigned long long int* d_found_results
) {
    // This kernel will perform a binary search for the target_fp
    // Only thread 0 needs to perform the search

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Check bloom filter first
        if (!check_bloom_filter(d_bloom_filter, bloom_size, target_fp)) {
            // Not in bloom filter, likely not in trap table
            return;
        }

        // Binary search in trap table
        int left = 0, right = trap_table_size - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            uint64_t mid_fp = d_trap_table[mid].fp;

            if (mid_fp == target_fp) {
                // Found match - reconstruct private key target
                uint64_t k_trap = d_trap_table[mid].privkey;
                unsigned long long int k_target = (unsigned long long int)((__int128)k_trap * pembagi);

                // Store result if k_target is valid (not zero)
                if (k_target != 0) {
                     // Use atomicExch to store the result (only one thread runs this)
                     atomicExch(&d_found_results[0], k_target);
                     atomicExch(&d_found_results[1], (unsigned long long int)k_trap);
                     // d_found_results[2] is not used in this version
                }
                return; // Exit after finding a match
            } else if (mid_fp < target_fp) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
}
