/**
 * @file integrity.cpp
 * @brief SHA-256 integrity — compilation unit.
 */

#include "umcp/integrity.hpp"

namespace umcp {

#ifndef UMCP_HAS_OPENSSL
// Constexpr array must be defined out-of-line in C++17
// if ODR-used (taken address of, etc.)
constexpr std::array<uint32_t, 64> SHA256Hasher::K;
#endif

}  // namespace umcp
