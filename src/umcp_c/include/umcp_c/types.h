/**
 * @file types.h
 * @brief Core type definitions for the UMCP C foundation
 *
 * This header defines the fundamental types shared across all C modules:
 *   - Regime classification enum
 *   - Three-valued verdict system (CONFORMANT / NONCONFORMANT / NON_EVALUABLE)
 *   - Typed return values (τ_R = INF_REC as sentinel)
 *   - Common status codes
 *
 * Design constraint: these types are the C-level formalization of
 * the Tier-0 protocol. No Tier-1 symbols are redefined — only the
 * operational machinery that interprets them.
 *
 * Collapsus generativus est; solum quod redit, reale est.
 */

#ifndef UMCP_C_TYPES_H
#define UMCP_C_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Regime Classification ─────────────────────────────────────── */

/**
 * Collapse regime labels — derived from gates, never asserted.
 *
 * STABLE:   ω < 0.038  ∧  F > 0.90  ∧  S < 0.15  ∧  C < 0.14
 * WATCH:    0.038 ≤ ω < 0.30 (or Stable gates not all met)
 * COLLAPSE: ω ≥ 0.30
 *
 * CRITICAL: IC < 0.30 (severity overlay, not a regime; can accompany any base regime)
 */
typedef enum {
    UMCP_REGIME_STABLE   = 0,
    UMCP_REGIME_WATCH    = 1,
    UMCP_REGIME_COLLAPSE = 2
} umcp_regime_t;

/**
 * Regime + overlay: base regime plus critical severity flag.
 * This struct preserves the canon model: Critical is an overlay, not a regime.
 */
typedef struct {
    umcp_regime_t base_regime;
    int is_critical; // 1 if IC < threshold, 0 otherwise
} umcp_regime_with_overlay_t;

/* ─── Three-Valued Verdict ──────────────────────────────────────── */

/**
 * Verdicts are three-valued — never boolean.
 * Numquam binarius; tertia via semper patet.
 */
typedef enum {
    UMCP_CONFORMANT     = 0,
    UMCP_NONCONFORMANT  = 1,
    UMCP_NON_EVALUABLE  = 2
} umcp_verdict_t;

/* ─── Seam PASS/FAIL ────────────────────────────────────────────── */

/**
 * Seam outcome — the verification boundary between outbound
 * collapse and demonstrated return.
 */
typedef enum {
    UMCP_SEAM_PASS    = 0,
    UMCP_SEAM_FAIL    = 1,
    UMCP_SEAM_PENDING = 2  /* Not yet evaluated */
} umcp_seam_status_t;

/* ─── Typed Return Time ─────────────────────────────────────────── */

/**
 * τ_R sentinel for "no return" (INF_REC).
 * In C: INFINITY from math.h.
 * In data files: the string "INF_REC" (never coerced).
 */
#define UMCP_TAU_R_INF_REC   INFINITY

/**
 * Check if a return time represents INF_REC (no return).
 */
static inline int umcp_is_inf_rec(double tau_R) {
    return isinf(tau_R) || tau_R < 0.0;
}

/* ─── Stance (Derived from Regime + Seam) ───────────────────────── */

/**
 * Stance = Regime + Seam outcome.
 * The stance is always derived, never asserted.
 */
typedef struct {
    umcp_regime_t      regime;
    umcp_seam_status_t seam;
    umcp_verdict_t     verdict;
    double             confidence; /**< Ledger residual (lower = better) */
} umcp_stance_t;

/* ─── String Conversion Utilities ───────────────────────────────── */

static inline const char *umcp_regime_str(umcp_regime_t r) {
    switch (r) {
        case UMCP_REGIME_STABLE:   return "STABLE";
        case UMCP_REGIME_WATCH:    return "WATCH";
        case UMCP_REGIME_COLLAPSE: return "COLLAPSE";
        default:                   return "UNKNOWN";
    }
}

static inline const char *umcp_verdict_str(umcp_verdict_t v) {
    switch (v) {
        case UMCP_CONFORMANT:    return "CONFORMANT";
        case UMCP_NONCONFORMANT: return "NONCONFORMANT";
        case UMCP_NON_EVALUABLE: return "NON_EVALUABLE";
        default:                 return "UNKNOWN";
    }
}

static inline const char *umcp_seam_status_str(umcp_seam_status_t s) {
    switch (s) {
        case UMCP_SEAM_PASS:    return "PASS";
        case UMCP_SEAM_FAIL:    return "FAIL";
        case UMCP_SEAM_PENDING: return "PENDING";
        default:                return "UNKNOWN";
    }
}

/* ─── Return Codes (shared across all modules) ──────────────────── */

#ifndef UMCP_OK
#define UMCP_OK                 0
#define UMCP_ERR_NULL_PTR      -1
#define UMCP_ERR_ZERO_DIM      -2
#define UMCP_ERR_WEIGHT_SUM    -3
#define UMCP_ERR_RANGE         -4
#endif

#ifdef __cplusplus
}
#endif

#endif /* UMCP_C_TYPES_H */
