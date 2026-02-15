/**
 * @file seam.hpp
 * @brief Seam Chain Accumulation — C++ Accelerator
 *
 * Tier-0 Protocol: incremental seam chain accounting (OPT-10, Lemma 20)
 * and residual growth monitoring (OPT-11, Lemma 27).
 *
 * Implements:
 *   - Lemma 20: Δκ_ledger composes additively across seam chains (O(1) query)
 *   - Lemma 27: Sublinear residual growth indicates returning dynamics
 *   - Lemma 19: Residual sensitivity to parameter perturbations
 *
 * No Tier-1 symbol is redefined. All frozen parameters come from the contract.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace umcp {

/**
 * Individual seam record with residual information.
 */
struct SeamRecord {
    int t0;                   ///< Start timestep
    int t1;                   ///< End timestep
    double kappa_t0;          ///< Log-integrity at t0
    double kappa_t1;          ///< Log-integrity at t1
    double tau_R;             ///< Return time
    double delta_kappa_ledger;///< Observed ledger change
    double delta_kappa_budget;///< Expected budget change
    double residual;          ///< Budget − ledger
    double cumulative_residual; ///< Running Σ|sₖ|
};

/**
 * Seam chain metrics summary.
 */
struct SeamChainMetrics {
    std::size_t total_seams;
    double total_delta_kappa;
    double cumulative_abs_residual;
    double max_residual;
    double mean_residual;
    double growth_exponent;
    bool is_returning;
    bool failure_detected;
};

/**
 * High-performance seam chain accumulator (OPT-10, OPT-11).
 *
 * O(1) incremental updates instead of O(K) recomputation.
 * Detects non-returning dynamics via sublinear growth test.
 */
class SeamChainAccumulator {
public:
    /**
     * @param alpha  Significance level for growth test
     * @param K_max  Maximum chain length before warning
     */
    explicit SeamChainAccumulator(double alpha = 0.05, std::size_t K_max = 1000)
        : alpha_(alpha)
        , K_max_(K_max)
        , total_delta_kappa_(0.0)
        , cumulative_abs_residual_(0.0)
        , failure_detected_(false)
    {}

    /**
     * Add a seam to the chain with O(1) incremental update (OPT-10).
     *
     * @param t0, t1  Seam endpoints
     * @param kappa_t0, kappa_t1  Log-integrity values
     * @param tau_R   Return time
     * @param R       Budget rate (return reward)
     * @param D_omega ω penalty term
     * @param D_C     C penalty term
     * @return SeamRecord with computed residual
     * @throws std::runtime_error if non-returning dynamics detected (OPT-11)
     */
    SeamRecord add_seam(int t0, int t1,
                        double kappa_t0, double kappa_t1,
                        double tau_R,
                        double R = 0.01,
                        double D_omega = 0.0,
                        double D_C = 0.0) {
        // Lemma 20: Ledger change composes additively
        double dk_ledger = kappa_t1 - kappa_t0;

        // Budget model (KERNEL_SPECIFICATION.md §3)
        double dk_budget = R * tau_R - (D_omega + D_C);

        // Residual
        double residual = dk_budget - dk_ledger;

        // OPT-10: O(1) incremental accumulation
        total_delta_kappa_ += dk_ledger;
        residuals_.push_back(residual);
        cumulative_abs_residual_ += std::abs(residual);

        SeamRecord record{};
        record.t0 = t0;
        record.t1 = t1;
        record.kappa_t0 = kappa_t0;
        record.kappa_t1 = kappa_t1;
        record.tau_R = tau_R;
        record.delta_kappa_ledger = dk_ledger;
        record.delta_kappa_budget = dk_budget;
        record.residual = residual;
        record.cumulative_residual = cumulative_abs_residual_;

        history_.push_back(record);

        // OPT-11: Check for failure every 10 seams
        if (history_.size() > 10 && history_.size() % 10 == 0) {
            check_residual_growth();
        }

        if (failure_detected_) {
            throw std::runtime_error(
                "Residual accumulation failure at K=" +
                std::to_string(history_.size()) +
                ". Non-returning dynamics (linear/superlinear growth).");
        }

        return record;
    }

    /** OPT-10: O(1) query for total ledger change (vs O(K) recomputation). */
    double total_delta_kappa() const { return total_delta_kappa_; }

    std::size_t size() const { return history_.size(); }

    bool failure_detected() const { return failure_detected_; }

    /** Compute comprehensive metrics. */
    SeamChainMetrics get_metrics() const {
        if (history_.empty()) {
            return {0, 0.0, 0.0, 0.0, 0.0, 0.0, false, false};
        }

        double max_res = 0.0;
        double sum_abs = 0.0;
        for (double r : residuals_) {
            double a = std::abs(r);
            sum_abs += a;
            if (a > max_res) max_res = a;
        }

        double mean_res = sum_abs / static_cast<double>(residuals_.size());
        double growth = compute_growth_exponent();

        return {
            history_.size(),
            total_delta_kappa_,
            cumulative_abs_residual_,
            max_res,
            mean_res,
            growth,
            growth < 1.05,  // Sublinear → returning
            failure_detected_
        };
    }

    /** Access the full residual history (for analysis). */
    const std::vector<double>& residuals() const { return residuals_; }

    /** Access the full seam history. */
    const std::vector<SeamRecord>& history() const { return history_; }

private:
    double alpha_;
    std::size_t K_max_;
    double total_delta_kappa_;
    double cumulative_abs_residual_;
    bool failure_detected_;
    std::vector<double> residuals_;
    std::vector<SeamRecord> history_;

    /**
     * OPT-11: Residual growth monitoring (Lemma 27).
     * Fit log(cumsum(|sₖ|)) ~ b·log(K).  b < 1 → sublinear (returning).
     */
    void check_residual_growth() {
        if (residuals_.size() < 10) return;
        double b = compute_growth_exponent();
        if (b > 1.05) {
            failure_detected_ = true;
        }
    }

    /**
     * Compute growth exponent b from cumsum ~ K^b.
     * Uses Welford-style online linear regression in log-log space.
     */
    double compute_growth_exponent() const {
        std::size_t K = residuals_.size();
        if (K < 10) return 0.0;

        // Cumulative sum of |residuals|
        std::vector<double> cumsum(K);
        cumsum[0] = std::abs(residuals_[0]);
        for (std::size_t i = 1; i < K; ++i) {
            cumsum[i] = cumsum[i - 1] + std::abs(residuals_[i]);
        }

        // Linear regression: log(cumsum) = b·log(k) + a
        // Using normal equations for speed
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        double n = static_cast<double>(K);
        for (std::size_t i = 0; i < K; ++i) {
            double x = std::log(static_cast<double>(i + 1));
            double y = std::log(cumsum[i] + 1e-10);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }

        double denom = n * sxx - sx * sx;
        if (std::abs(denom) < 1e-15) return 0.0;

        double b = (n * sxy - sx * sy) / denom;
        return b;
    }
};

/**
 * Residual sensitivity calculator (Lemma 19).
 * Computes partial derivatives of residual w.r.t. each parameter.
 */
struct ResidualSensitivity {
    double ds_dR;           ///< ∂s/∂R = τ_R
    double ds_dtau_R;       ///< ∂s/∂τ_R = R
    double ds_dD_omega;     ///< ∂s/∂D_ω = −1
    double ds_dD_C;         ///< ∂s/∂D_C = −1
    double ds_dkappa_ledger;///< ∂s/∂κ_ledger = −1

    static ResidualSensitivity compute(double tau_R, double R) {
        return {tau_R, R, -1.0, -1.0, -1.0};
    }
};

}  // namespace umcp
