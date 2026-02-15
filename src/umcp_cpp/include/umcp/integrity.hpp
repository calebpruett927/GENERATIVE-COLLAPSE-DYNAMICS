/**
 * @file integrity.hpp
 * @brief SHA-256 File Integrity — C++ Accelerator
 *
 * Tier-0 Protocol: SHA-256 checksum computation for integrity verification.
 * Uses OpenSSL if available, falls back to a portable C++ implementation.
 *
 * This is the I/O-bound operation where C++ gains ~5× over Python's hashlib
 * due to larger read buffers and zero-copy processing.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef UMCP_HAS_OPENSSL
#include <openssl/sha.h>
#endif

namespace umcp {

/**
 * SHA-256 hasher.
 * Uses OpenSSL EVP if available, otherwise a built-in implementation.
 */
class SHA256Hasher {
public:
    /// Size of a SHA-256 digest in bytes
    static constexpr std::size_t DIGEST_SIZE = 32;

    /// Read buffer size (256 KB for optimal I/O throughput)
    static constexpr std::size_t BUFFER_SIZE = 256 * 1024;

    /**
     * Compute SHA-256 hash of a file.
     *
     * @param filepath Path to the file
     * @return Hex-encoded SHA-256 digest (64 characters)
     * @throws std::runtime_error if file cannot be opened
     */
    static std::string hash_file(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        std::array<unsigned char, DIGEST_SIZE> digest{};

#ifdef UMCP_HAS_OPENSSL
        SHA256_CTX ctx;
        SHA256_Init(&ctx);

        std::vector<char> buffer(BUFFER_SIZE);
        while (file.read(buffer.data(), static_cast<std::streamsize>(BUFFER_SIZE)) || file.gcount() > 0) {
            SHA256_Update(&ctx, buffer.data(), static_cast<std::size_t>(file.gcount()));
        }

        SHA256_Final(digest.data(), &ctx);
#else
        // Portable SHA-256 implementation
        digest = sha256_portable(file);
#endif

        return to_hex(digest);
    }

    /**
     * Compute SHA-256 hash of a byte buffer.
     *
     * @param data Pointer to data
     * @param len  Length in bytes
     * @return Hex-encoded SHA-256 digest
     */
    static std::string hash_bytes(const void* data, std::size_t len) {
        std::array<unsigned char, DIGEST_SIZE> digest{};

#ifdef UMCP_HAS_OPENSSL
        SHA256(static_cast<const unsigned char*>(data), len, digest.data());
#else
        digest = sha256_bytes_portable(
            static_cast<const unsigned char*>(data), len);
#endif

        return to_hex(digest);
    }

    /**
     * Verify a file against an expected hash.
     *
     * @param filepath Path to the file
     * @param expected_hash Expected hex-encoded SHA-256 digest
     * @return true if hashes match
     */
    static bool verify_file(const std::string& filepath,
                            const std::string& expected_hash) {
        return hash_file(filepath) == expected_hash;
    }

    /**
     * Batch hash multiple files.
     *
     * @param filepaths Vector of file paths
     * @return Vector of (filepath, hash) pairs
     */
    static std::vector<std::pair<std::string, std::string>>
    hash_files(const std::vector<std::string>& filepaths) {
        std::vector<std::pair<std::string, std::string>> results;
        results.reserve(filepaths.size());
        for (const auto& fp : filepaths) {
            results.emplace_back(fp, hash_file(fp));
        }
        return results;
    }

private:
    static std::string to_hex(const std::array<unsigned char, DIGEST_SIZE>& digest) {
        std::ostringstream oss;
        for (unsigned char byte : digest) {
            oss << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(byte);
        }
        return oss.str();
    }

#ifndef UMCP_HAS_OPENSSL
    // ─── Portable SHA-256 (RFC 6234) ────────────────────────────────
    // Self-contained implementation for environments without OpenSSL.

    static constexpr std::array<uint32_t, 64> K = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    static uint32_t rotr(uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }

    static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    static uint32_t sigma0(uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }

    static uint32_t sigma1(uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }

    static uint32_t gamma0(uint32_t x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }

    static uint32_t gamma1(uint32_t x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    struct SHA256State {
        std::array<uint32_t, 8> h;
        uint64_t total_len;
        std::array<unsigned char, 64> buffer;
        std::size_t buffer_len;

        SHA256State() : total_len(0), buffer{}, buffer_len(0) {
            h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        }
    };

    static void process_block(SHA256State& state, const unsigned char* block) {
        std::array<uint32_t, 64> W{};

        // Message schedule
        for (int i = 0; i < 16; ++i) {
            W[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(block[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            W[i] = gamma1(W[i - 2]) + W[i - 7] + gamma0(W[i - 15]) + W[i - 16];
        }

        uint32_t a = state.h[0], b = state.h[1], c = state.h[2], d = state.h[3];
        uint32_t e = state.h[4], f = state.h[5], g = state.h[6], hh = state.h[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t T1 = hh + sigma1(e) + ch(e, f, g) + K[i] + W[i];
            uint32_t T2 = sigma0(a) + maj(a, b, c);
            hh = g; g = f; f = e; e = d + T1;
            d = c; c = b; b = a; a = T1 + T2;
        }

        state.h[0] += a; state.h[1] += b; state.h[2] += c; state.h[3] += d;
        state.h[4] += e; state.h[5] += f; state.h[6] += g; state.h[7] += hh;
    }

    static void sha256_update(SHA256State& state,
                              const unsigned char* data, std::size_t len) {
        state.total_len += len;
        std::size_t offset = 0;

        // Fill buffer
        if (state.buffer_len > 0) {
            std::size_t space = 64 - state.buffer_len;
            std::size_t fill = (len < space) ? len : space;
            for (std::size_t i = 0; i < fill; ++i)
                state.buffer[state.buffer_len + i] = data[i];
            state.buffer_len += fill;
            offset += fill;

            if (state.buffer_len == 64) {
                process_block(state, state.buffer.data());
                state.buffer_len = 0;
            }
        }

        // Process full blocks
        while (offset + 64 <= len) {
            process_block(state, data + offset);
            offset += 64;
        }

        // Buffer remainder
        std::size_t remaining = len - offset;
        for (std::size_t i = 0; i < remaining; ++i)
            state.buffer[i] = data[offset + i];
        state.buffer_len = remaining;
    }

    static std::array<unsigned char, DIGEST_SIZE> sha256_final(SHA256State& state) {
        // Padding
        uint64_t bit_len = state.total_len * 8;
        unsigned char pad = 0x80;
        sha256_update(state, &pad, 1);

        unsigned char zero = 0x00;
        while (state.buffer_len != 56) {
            sha256_update(state, &zero, 1);
        }

        // Append length in big-endian
        unsigned char len_bytes[8];
        for (int i = 7; i >= 0; --i) {
            len_bytes[i] = static_cast<unsigned char>(bit_len & 0xFF);
            bit_len >>= 8;
        }
        sha256_update(state, len_bytes, 8);

        // Extract digest
        std::array<unsigned char, DIGEST_SIZE> digest{};
        for (int i = 0; i < 8; ++i) {
            digest[i * 4]     = static_cast<unsigned char>((state.h[i] >> 24) & 0xFF);
            digest[i * 4 + 1] = static_cast<unsigned char>((state.h[i] >> 16) & 0xFF);
            digest[i * 4 + 2] = static_cast<unsigned char>((state.h[i] >> 8) & 0xFF);
            digest[i * 4 + 3] = static_cast<unsigned char>(state.h[i] & 0xFF);
        }
        return digest;
    }

    static std::array<unsigned char, DIGEST_SIZE>
    sha256_portable(std::ifstream& file) {
        SHA256State state;
        std::vector<unsigned char> buffer(BUFFER_SIZE);
        while (file.read(reinterpret_cast<char*>(buffer.data()),
                         static_cast<std::streamsize>(BUFFER_SIZE)) ||
               file.gcount() > 0) {
            sha256_update(state, buffer.data(),
                          static_cast<std::size_t>(file.gcount()));
        }
        return sha256_final(state);
    }

    static std::array<unsigned char, DIGEST_SIZE>
    sha256_bytes_portable(const unsigned char* data, std::size_t len) {
        SHA256State state;
        sha256_update(state, data, len);
        return sha256_final(state);
    }
#endif  // !UMCP_HAS_OPENSSL
};

}  // namespace umcp
