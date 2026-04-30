#ifndef LIBNQX_H
#define LIBNQX_H

/* NQX-Core C ABI — prototype header for libnqx.
 *
 * Compile with: gcc test.c -lnqx (or link against sdk/libnqx/libnqx.py via ctypes)
 *
 * Current implementation is a Python prototype. Real .so coming via pybind11/ctypes.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to an NQX accelerator instance. */
typedef void* nqx_handle;

/* Encode result (caller frees arrays with free()) */
typedef struct {
    uint8_t* packed;      /* packed byte stream */
    size_t    packed_len;
    uint8_t* sign_bits;   /* sign correction bits */
    size_t    sign_len;
    float*   mins;        /* per-dimension min */
    float*   maxs;        /* per-dimension max */
    size_t    dim;
    uint32_t  n;           /* number of vectors */
    uint32_t  bits;
    double    encode_ms;
    double    rmse;
} nqx_encode_result;

/* Decode result */
typedef struct {
    float*  vectors;      /* [n x dim] reconstructed */
    size_t   n;
    size_t   dim;
    double   decode_ms;
    double   rmse;
} nqx_decode_result;

/* Open accelerator handle.
 * config: JSON string, e.g. '{"dim":128,"bits":3}'
 * Returns NULL on failure.
 */
nqx_handle nqx_open(const char* config);

/* Encode vectors.
 * handle: from nqx_open()
 * vectors: flat [n * dim] float array
 * n: number of vectors
 * dim: dimension per vector
 * bits: quantization bits (0 = default)
 * out: caller-allocated nqx_encode_result
 * Returns 0 on success, -1 on error.
 */
int nqx_encode(nqx_handle handle, const float* vectors, size_t n, size_t dim,
               uint32_t bits, nqx_encode_result* out);

/* Decode packed data.
 * handle: from nqx_open()
 * packed: packed byte stream
 * packed_len: byte count
 * sign_bits: sign correction bits
 * sign_len: byte count
 * mins, maxs: per-dimension float arrays
 * n, dim: vector count and dimension
 * bits: quantization bits
 * out: caller-allocated nqx_decode_result
 * Returns 0 on success, -1 on error.
 */
int nqx_decode(nqx_handle handle, const uint8_t* packed, size_t packed_len,
               const uint8_t* sign_bits, size_t sign_len,
               const float* mins, const float* maxs,
               size_t n, size_t dim, uint32_t bits,
               nqx_decode_result* out);

/* Free encode result internals (allocated by libnqx). */
void nqx_free_encode_result(nqx_encode_result* out);

/* Free decode result internals (allocated by libnqx). */
void nqx_free_decode_result(nqx_decode_result* out);

/* Close handle and free resources. */
void nqx_close(nqx_handle handle);

/* Version string. */
const char* nqx_version(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBNQX_H */
