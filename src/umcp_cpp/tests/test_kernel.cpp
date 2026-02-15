/**
 * UMCP C++ Kernel Tests — Catch2 v3
 *
 * Tier-0 Protocol: Verifies the C++ implementation produces
 * results consistent with Tier-1 identities.
 *
 * Tests mirror the Python test suite's Tier-1 checks:
 *   - F + ω = 1  (Complementum Perfectum)
 *   - IC ≤ F     (Limbus Integritatis / integrity bound)
 *   - IC = exp(κ) (Log-integrity relation)
 *   - Ranges     (Lemma 1: F, IC ∈ [ε, 1−ε], S ≥ 0, C ≥ 0)
 *
 * Build:
 *   cd src/umcp_cpp && mkdir build && cd build
 *   cmake .. -DUMCP_BUILD_TESTS=ON && make
 *   ./umcp_tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "umcp/kernel.hpp"
#include "umcp/seam.hpp"
#include "umcp/integrity.hpp"

#include <cmath>
#include <vector>
#include <numeric>
#include <random>

using Catch::Approx;
using namespace umcp;

// ─────────────────────── Helpers ──────────────────────────────────

static std::vector<double> uniform_weights(size_t n) {
    return std::vector<double>(n, 1.0 / static_cast<double>(n));
}

static std::vector<double> constant_coords(size_t n, double val) {
    return std::vector<double>(n, val);
}

// ═══════════════════ Kernel: Tier‑1 Identities ═══════════════════

TEST_CASE("Duality identity F + omega = 1", "[kernel][tier1]") {
    KernelComputer kern;

    SECTION("Uniform coordinates") {
        auto c = constant_coords(8, 0.75);
        auto w = uniform_weights(8);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.F + out.omega == Approx(1.0).epsilon(1e-14));
    }

    SECTION("Random 64-channel") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.1, 0.9);
        std::vector<double> c(64), w = uniform_weights(64);
        for (auto& ci : c) ci = dist(rng);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.F + out.omega == Approx(1.0).epsilon(1e-14));
    }

    SECTION("Near-epsilon coordinates") {
        double eps = 1e-8;
        auto c = constant_coords(8, eps);
        auto w = uniform_weights(8);
        KernelComputer kern2(eps);
        auto out = kern2.compute(c.data(), w.data(), c.size());
        REQUIRE(out.F + out.omega == Approx(1.0).epsilon(1e-14));
    }

    SECTION("Near unity") {
        double eps = 1e-8;
        auto c = constant_coords(8, 1.0 - eps);
        auto w = uniform_weights(8);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.F + out.omega == Approx(1.0).epsilon(1e-14));
    }
}

TEST_CASE("Integrity bound IC <= F", "[kernel][tier1]") {
    KernelComputer kern;

    SECTION("Uniform — IC = F when homogeneous") {
        auto c = constant_coords(32, 0.6);
        auto w = uniform_weights(32);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.IC <= out.F + 1e-14);
        // Homogeneous → gap is zero
        REQUIRE(out.delta == Approx(0.0).margin(1e-14));
    }

    SECTION("Heterogeneous — IC < F") {
        std::vector<double> c = {0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4};
        auto w = uniform_weights(8);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.IC < out.F);
        REQUIRE(out.delta > 0.0);
    }

    SECTION("Sweep 100 random vectors") {
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(0.05, 0.95);
        for (int trial = 0; trial < 100; ++trial) {
            std::vector<double> c(16);
            for (auto& ci : c) ci = dist(rng);
            auto w = uniform_weights(16);
            auto out = kern.compute(c.data(), w.data(), c.size());
            REQUIRE(out.IC <= out.F + 1e-14);
        }
    }
}

TEST_CASE("Log-integrity IC = exp(kappa)", "[kernel][tier1]") {
    KernelComputer kern;

    SECTION("Exact for homogeneous") {
        auto c = constant_coords(8, 0.5);
        auto w = uniform_weights(8);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.IC == Approx(std::exp(out.kappa)).epsilon(1e-14));
    }

    SECTION("Random heterogeneous") {
        std::mt19937 rng(77);
        std::uniform_real_distribution<double> dist(0.1, 0.9);
        std::vector<double> c(32);
        for (auto& ci : c) ci = dist(rng);
        auto w = uniform_weights(32);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.IC == Approx(std::exp(out.kappa)).epsilon(1e-12));
    }
}

// ═══════════════════ Kernel: Range Validation ═══════════════════

TEST_CASE("Kernel output ranges (Lemma 1)", "[kernel][ranges]") {
    KernelComputer kern;
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(0.01, 0.99);

    for (int n : {4, 8, 16, 64, 256}) {
        SECTION("n = " + std::to_string(n)) {
            std::vector<double> c(n);
            for (auto& ci : c) ci = dist(rng);
            auto w = uniform_weights(n);
            auto out = kern.compute(c.data(), w.data(), c.size());

            REQUIRE(out.F >= 0.0);
            REQUIRE(out.F <= 1.0);
            REQUIRE(out.omega >= 0.0);
            REQUIRE(out.omega <= 1.0);
            REQUIRE(out.S >= 0.0);
            REQUIRE(out.C >= 0.0);
            REQUIRE(out.IC >= 0.0);
            REQUIRE(out.IC <= 1.0);
        }
    }
}

// ══════════════════ Kernel: Homogeneous Detection ═══════════════

TEST_CASE("Homogeneous path (OPT-1)", "[kernel][opt]") {
    KernelComputer kern;

    SECTION("Perfectly uniform triggers fast path") {
        auto c = constant_coords(256, 0.75);
        auto w = uniform_weights(256);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.is_homogeneous == true);
        REQUIRE(out.computation_mode == "fast_homogeneous");
        REQUIRE(out.F == Approx(0.75).epsilon(1e-14));
        REQUIRE(out.IC == Approx(0.75).epsilon(1e-14));
        REQUIRE(out.delta == Approx(0.0).margin(1e-14));
    }

    SECTION("Heterogeneous uses full path") {
        std::vector<double> c = {0.1, 0.9, 0.5, 0.3};
        auto w = uniform_weights(4);
        auto out = kern.compute(c.data(), w.data(), c.size());
        REQUIRE(out.is_homogeneous == false);
        REQUIRE(out.computation_mode == "full_heterogeneous");
    }
}

// ══════════════════ Kernel: Batch Computation ════════════════════

TEST_CASE("Batch computation matches single", "[kernel][batch]") {
    KernelComputer kern;
    std::mt19937 rng(55);
    std::uniform_real_distribution<double> dist(0.1, 0.9);

    const size_t T = 50, n = 8;
    std::vector<double> trace(T * n);
    for (auto& v : trace) v = dist(rng);
    auto w = uniform_weights(n);

    auto batch = kern.compute_batch(trace.data(), w.data(), T, n);

    for (size_t t = 0; t < T; ++t) {
        auto single = kern.compute(trace.data() + t * n, w.data(), n);
        REQUIRE(batch.F[t] == Approx(single.F).epsilon(1e-14));
        REQUIRE(batch.omega[t] == Approx(single.omega).epsilon(1e-14));
        REQUIRE(batch.S[t] == Approx(single.S).epsilon(1e-12));
        REQUIRE(batch.kappa[t] == Approx(single.kappa).epsilon(1e-14));
        REQUIRE(batch.IC[t] == Approx(single.IC).epsilon(1e-14));
    }
}

// ═══════════════════ Kernel: Error Propagation ═══════════════════

TEST_CASE("Error propagation (OPT-12, Lemma 23)", "[kernel][error]") {
    KernelComputer kern;
    double eps = 1e-8;
    auto bounds = kern.propagate_error(1e-4, eps);

    REQUIRE(bounds.delta_F == Approx(1e-4));
    REQUIRE(bounds.delta_omega == Approx(1e-4));
    REQUIRE(bounds.delta_kappa > 0.0);
    REQUIRE(bounds.delta_S > 0.0);
    // Lipschitz constant for kappa is 1/epsilon
    REQUIRE(bounds.delta_kappa == Approx(1e-4 / eps).epsilon(1e-10));
}

// ═══════════════════ Seam Chain ═════════════════════════════════

TEST_CASE("SeamChainAccumulator basic operations", "[seam]") {
    SeamChainAccumulator chain(0.05, 100);

    SECTION("Empty chain") {
        REQUIRE(chain.size() == 0);
        REQUIRE(chain.total_delta_kappa() == Approx(0.0));
        REQUIRE(chain.failure_detected() == false);
    }

    SECTION("Single seam") {
        auto rec = chain.add_seam(0, 1, -0.5, -0.48, 5.0, 0.01, 0.001, 0.0005);
        REQUIRE(chain.size() == 1);
        REQUIRE(rec.delta_kappa_ledger == Approx(-0.48 - (-0.5)).epsilon(1e-14));
    }

    SECTION("O(1) accumulation") {
        double kappa = -1.0;
        for (int k = 0; k < 500; ++k) {
            double next_kappa = kappa + 0.001;
            chain.add_seam(k, k + 1, kappa, next_kappa, 5.0, 0.01, 0.001, 0.0005);
            kappa = next_kappa;
        }
        REQUIRE(chain.size() == 500);
        REQUIRE(chain.total_delta_kappa() == Approx(0.5).margin(1e-10));
    }
}

TEST_CASE("SeamChainAccumulator metrics", "[seam][metrics]") {
    SeamChainAccumulator chain(0.05, 1000);
    std::mt19937 rng(42);
    std::normal_distribution<double> dk_dist(0.0, 0.001);

    double kappa = -0.5;
    for (int k = 0; k < 200; ++k) {
        double next = kappa + dk_dist(rng);
        chain.add_seam(k, k + 1, kappa, next, 3.0, 0.01, 0.002, 0.001);
        kappa = next;
    }

    auto metrics = chain.get_metrics();
    REQUIRE(metrics.total_seams == 200);
    REQUIRE(std::isfinite(metrics.total_delta_kappa));
    REQUIRE(metrics.cumulative_abs_residual >= 0.0);
    REQUIRE(metrics.max_residual >= 0.0);
}

// ═══════════════════ SHA-256 Integrity ══════════════════════════

TEST_CASE("SHA-256 known test vectors", "[integrity]") {
    SHA256Hasher hasher;

    SECTION("Empty string") {
        auto h = hasher.hash_bytes(std::vector<uint8_t>{});
        REQUIRE(h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    SECTION("'abc'") {
        std::vector<uint8_t> data = {'a', 'b', 'c'};
        auto h = hasher.hash_bytes(data);
        REQUIRE(h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    SECTION("'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'") {
        std::string s = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        std::vector<uint8_t> data(s.begin(), s.end());
        auto h = hasher.hash_bytes(data);
        REQUIRE(h == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }
}

TEST_CASE("SHA-256 file hashing", "[integrity]") {
    SHA256Hasher hasher;

    // Create temp file
    const char* tmp = "/tmp/umcp_test_sha256.dat";
    {
        FILE* f = fopen(tmp, "wb");
        REQUIRE(f != nullptr);
        const char* data = "Hello, UMCP!";
        fwrite(data, 1, 12, f);
        fclose(f);
    }

    auto h = hasher.hash_file(tmp);
    REQUIRE(h.size() == 64);  // 256 bits = 64 hex chars

    // Verify matches hash_bytes
    std::vector<uint8_t> data = {'H','e','l','l','o',',',' ','U','M','C','P','!'};
    REQUIRE(h == hasher.hash_bytes(data));

    // Verify
    REQUIRE(hasher.verify_file(tmp, h) == true);
    REQUIRE(hasher.verify_file(tmp, "0000000000000000000000000000000000000000000000000000000000000000") == false);

    remove(tmp);
}

// ═══════════════ Regime Classification ══════════════════════════

TEST_CASE("Regime classification", "[kernel][regime]") {
    KernelComputer kern;

    SECTION("Stable: low omega") {
        auto c = constant_coords(8, 0.9);  // F ≈ 0.9, ω ≈ 0.1
        auto w = uniform_weights(8);
        auto out = kern.compute(c.data(), w.data(), c.size());
        auto regime = kern.classify_regime(out);
        REQUIRE(regime == "Stable");
    }

    SECTION("Collapse: high omega") {
        auto c = constant_coords(8, 0.4);  // F ≈ 0.4, ω ≈ 0.6
        auto w = uniform_weights(8);
        auto out = kern.compute(c.data(), w.data(), c.size());
        auto regime = kern.classify_regime(out);
        REQUIRE(regime == "Collapse");
    }
}

// ══════════════════ Exhaustive Tier-1 Sweep ═════════════════════

TEST_CASE("10K random vectors: all Tier-1 identities hold", "[kernel][tier1][sweep]") {
    KernelComputer kern;
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> c_dist(0.01, 0.99);
    std::uniform_int_distribution<int> n_dist(4, 64);

    int violations = 0;

    for (int trial = 0; trial < 10000; ++trial) {
        size_t n = static_cast<size_t>(n_dist(rng));
        std::vector<double> c(n);
        for (auto& ci : c) ci = c_dist(rng);
        auto w = uniform_weights(n);

        auto out = kern.compute(c.data(), w.data(), n);

        // F + ω = 1
        if (std::abs(out.F + out.omega - 1.0) > 1e-14) ++violations;
        // IC ≤ F
        if (out.IC > out.F + 1e-14) ++violations;
        // IC = exp(κ)
        if (std::abs(out.IC - std::exp(out.kappa)) > 1e-10) ++violations;
        // Ranges
        if (out.F < 0.0 || out.F > 1.0) ++violations;
        if (out.S < 0.0) ++violations;
        if (out.C < 0.0) ++violations;
    }

    REQUIRE(violations == 0);
}
