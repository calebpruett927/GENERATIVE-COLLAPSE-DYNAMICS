/**
 * @file regime.c
 * @brief Regime Classification implementation
 *
 * Regime labels are derived from frozen gates on Tier-1 invariants.
 * Diagnostica informant, portae decernunt.
 * (Diagnostics inform; gates decide.)
 */

#include "umcp_c/regime.h"
#include <stdlib.h>
#include <math.h>

/* ─── Regime Classification ─────────────────────────────────────── */


umcp_regime_with_overlay_t umcp_classify_regime_with_overlay(
    const umcp_kernel_result_t *result,
    const umcp_regime_thresholds_t *thresholds)
{
    umcp_regime_with_overlay_t out;
    out.is_critical = 0;
    if (!result || !thresholds) {
        out.base_regime = UMCP_REGIME_WATCH;
        return out;
    }

    double omega = result->omega;
    double F     = result->F;
    double S     = result->S;
    double C     = result->C;
    double IC    = result->IC;

    if (IC < thresholds->IC_critical_max) {
        out.is_critical = 1;
    }

    /* Collapse: ω ≥ 0.30 */
    if (omega >= thresholds->omega_collapse_min) {
        out.base_regime = UMCP_REGIME_COLLAPSE;
        return out;
    }

    /* Watch: 0.038 ≤ ω < 0.30 */
    if (omega >= thresholds->omega_watch_min) {
        out.base_regime = UMCP_REGIME_WATCH;
        return out;
    }

    /* Stable requires ALL conditions (conjunctive) */
    if (omega < thresholds->omega_stable_max &&
        F     > thresholds->F_stable_min &&
        S     < thresholds->S_stable_max &&
        C     < thresholds->C_stable_max) {
        out.base_regime = UMCP_REGIME_STABLE;
        return out;
    }

    /* Default to Watch if not clearly stable */
    out.base_regime = UMCP_REGIME_WATCH;
    return out;
}

// Legacy API for compatibility: returns only the base regime (no overlay)
umcp_regime_t umcp_classify_regime(
    const umcp_kernel_result_t *result,
    const umcp_regime_thresholds_t *thresholds)
{
    umcp_regime_with_overlay_t tmp = umcp_classify_regime_with_overlay(result, thresholds);
    return tmp.base_regime;
}

umcp_regime_t umcp_classify_regime_default(const umcp_kernel_result_t *result)
{
    umcp_regime_thresholds_t defaults = {
        .omega_stable_max   = 0.038,
        .F_stable_min       = 0.90,
        .S_stable_max       = 0.15,
        .C_stable_max       = 0.14,
        .omega_watch_min    = 0.038,
        .omega_watch_max    = 0.30,
        .omega_collapse_min = 0.30,
        .IC_critical_max    = 0.30
    };
    return umcp_classify_regime(result, &defaults);
}

int umcp_is_critical(const umcp_kernel_result_t *result,
                     const umcp_regime_thresholds_t *thresholds)
{
    if (!result || !thresholds) return 0;
    return result->IC < thresholds->IC_critical_max ? 1 : 0;
}

/* ─── Fisher-Space Partition ────────────────────────────────────── */

/*
 * Simple LCG for deterministic pseudo-random numbers.
 * Not cryptographic — used only for regime partition estimation.
 */
static uint64_t lcg_state = 0;

static void lcg_seed(uint64_t seed) { lcg_state = seed; }

static double lcg_uniform(void)
{
    /* Numerical Recipes LCG: period 2^64 */
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 11) / (double)(1ULL << 53);
}

void umcp_regime_partition(
    size_t n_channels, size_t n_samples,
    const umcp_regime_thresholds_t *thresholds,
    double *pct_stable, double *pct_watch,
    double *pct_collapse, double *pct_critical)
{
    if (!thresholds || !pct_stable || !pct_watch ||
        !pct_collapse || !pct_critical || n_channels == 0 || n_samples == 0) {
        return;
    }

    lcg_seed(42);  /* Deterministic for reproducibility */

    size_t counts[3] = {0, 0, 0};
    size_t count_critical = 0;
    double *c = (double *)malloc(n_channels * sizeof(double));
    double *w = (double *)malloc(n_channels * sizeof(double));
    if (!c || !w) {
        free(c); free(w);
        return;
    }

    double wval = 1.0 / (double)n_channels;
    for (size_t j = 0; j < n_channels; ++j) w[j] = wval;

    umcp_kernel_result_t result;
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_channels; ++j) {
            c[j] = lcg_uniform();
            /* Clamp to [ε, 1-ε] */
            if (c[j] < 1e-8) c[j] = 1e-8;
            if (c[j] > 1.0 - 1e-8) c[j] = 1.0 - 1e-8;
        }

        if (umcp_kernel_compute(c, w, n_channels, 1e-8, &result) == UMCP_OK) {
            umcp_regime_with_overlay_t regime = umcp_classify_regime_with_overlay(&result, thresholds);
            counts[regime.base_regime]++;
            if (regime.is_critical) count_critical++;
        }
    }

    double total = (double)n_samples;
    *pct_stable   = 100.0 * (double)counts[UMCP_REGIME_STABLE]   / total;
    *pct_watch    = 100.0 * (double)counts[UMCP_REGIME_WATCH]    / total;
    *pct_collapse = 100.0 * (double)counts[UMCP_REGIME_COLLAPSE] / total;
    *pct_critical = 100.0 * (double)count_critical / total;

    free(c);
    free(w);
}
