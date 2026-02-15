/**
 * @file kernel.cpp
 * @brief Kernel computation — compilation unit.
 *
 * The kernel is header-only for inlining in the hot path,
 * but this .cpp ensures the symbols are available for linking
 * and provides any non-inline utility functions.
 */

#include "umcp/kernel.hpp"

// Explicit template instantiations and non-inline definitions
// would go here.  Currently the kernel is fully header-inline
// for maximum performance in the pybind11 module.

namespace umcp {

// Reserved for future non-inline implementations
// (e.g., SIMD-specialized batch kernels for AVX-512)

}  // namespace umcp
