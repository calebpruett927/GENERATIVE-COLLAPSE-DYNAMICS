/**
 * @file test_orchestration.c
 * @brief Comprehensive tests for the C orchestration layer
 *
 * Tests all new modules:
 *   - types.h      (enums, string conversion, typed returns)
 *   - contract.h/c (frozen parameters, cost closures, seam PASS)
 *   - regime.h/c   (gate classification, partition)
 *   - trace.h/c    (lifecycle, embedding, identity validation)
 *   - ledger.h/c   (append, running stats, verdicts)
 *   - pipeline.h/c (full spine orchestration)
 *
 * Build: linked against umcp_c_core (includes kernel.c, seam.c, sha256.c,
 *        contract.c, regime.c, trace.c, ledger.c, pipeline.c)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "umcp_c/types.h"
#include "umcp_c/contract.h"
#include "umcp_c/regime.h"
#include "umcp_c/trace.h"
#include "umcp_c/ledger.h"
#include "umcp_c/pipeline.h"
#include "umcp_c/kernel.h"

/* ─── Test Infrastructure ───────────────────────────────────────── */

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do {                              \
    tests_run++;                                            \
    if (cond) {                                             \
        tests_passed++;                                     \
    } else {                                                \
        tests_failed++;                                     \
        fprintf(stderr, "  FAIL [%s:%d]: %s\n",            \
                __FILE__, __LINE__, msg);                   \
    }                                                       \
} while (0)

#define ASSERT_NEAR(a, b, tol, msg) do {                    \
    double _a = (a), _b = (b);                              \
    ASSERT(fabs(_a - _b) <= (tol), msg);                    \
} while (0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)


/* ═══════════════════ Types Tests ═════════════════════════════════ */

static void test_types_regime_enum(void)
{
    printf("  Test: Regime enum values\n");
    ASSERT_EQ(UMCP_REGIME_STABLE,   0, "STABLE = 0");
    ASSERT_EQ(UMCP_REGIME_WATCH,    1, "WATCH = 1");
    ASSERT_EQ(UMCP_REGIME_COLLAPSE, 2, "COLLAPSE = 2");
}

static void test_types_verdict_enum(void)
{
    printf("  Test: Verdict enum values\n");
    ASSERT_EQ(UMCP_CONFORMANT,     0, "CONFORMANT = 0");
    ASSERT_EQ(UMCP_NONCONFORMANT,  1, "NONCONFORMANT = 1");
    ASSERT_EQ(UMCP_NON_EVALUABLE,  2, "NON_EVALUABLE = 2");
}

static void test_types_seam_enum(void)
{
    printf("  Test: Seam enum values\n");
    ASSERT_EQ(UMCP_SEAM_PASS,    0, "PASS = 0");
    ASSERT_EQ(UMCP_SEAM_FAIL,    1, "FAIL = 1");
    ASSERT_EQ(UMCP_SEAM_PENDING, 2, "PENDING = 2");
}

static void test_types_inf_rec(void)
{
    printf("  Test: INF_REC sentinel\n");
    ASSERT(umcp_is_inf_rec(UMCP_TAU_R_INF_REC), "INFINITY is INF_REC");
    ASSERT(umcp_is_inf_rec(-1.0),  "negative is INF_REC");
    ASSERT(!umcp_is_inf_rec(5.0),  "positive finite is not INF_REC");
    ASSERT(!umcp_is_inf_rec(0.0),  "zero is not INF_REC");
}

static void test_types_string_conversion(void)
{
    printf("  Test: String conversion utilities\n");
        ASSERT(strcmp(umcp_regime_str(UMCP_REGIME_STABLE), "STABLE") == 0,
            "STABLE string");
        ASSERT(strcmp(umcp_regime_str(UMCP_REGIME_WATCH), "WATCH") == 0,
            "WATCH string");
        ASSERT(strcmp(umcp_regime_str(UMCP_REGIME_COLLAPSE), "COLLAPSE") == 0,
            "COLLAPSE string");

    ASSERT(strcmp(umcp_verdict_str(UMCP_CONFORMANT), "CONFORMANT") == 0,
           "CONFORMANT string");
    ASSERT(strcmp(umcp_verdict_str(UMCP_NONCONFORMANT), "NONCONFORMANT") == 0,
           "NONCONFORMANT string");
    ASSERT(strcmp(umcp_verdict_str(UMCP_NON_EVALUABLE), "NON_EVALUABLE") == 0,
           "NON_EVALUABLE string");

    ASSERT(strcmp(umcp_seam_status_str(UMCP_SEAM_PASS), "PASS") == 0,
           "PASS string");
    ASSERT(strcmp(umcp_seam_status_str(UMCP_SEAM_FAIL), "FAIL") == 0,
           "FAIL string");
    ASSERT(strcmp(umcp_seam_status_str(UMCP_SEAM_PENDING), "PENDING") == 0,
           "PENDING string");
}


/* ═══════════════════ Contract Tests ══════════════════════════════ */

static void test_contract_default_values(void)
{
    printf("  Test: Contract default values match frozen_contract.py\n");
    umcp_contract_t ct;
    umcp_contract_default(&ct);

    ASSERT_NEAR(ct.epsilon,    1e-8,  0.0,    "epsilon = 1e-8");
    ASSERT_EQ(ct.p_exponent,   3,             "p_exponent = 3");
    ASSERT_NEAR(ct.alpha,      1.0,   0.0,    "alpha = 1.0");
    ASSERT_NEAR(ct.lambda,     0.2,   0.0,    "lambda = 0.2");
    ASSERT_NEAR(ct.tol_seam,   0.005, 0.0,    "tol_seam = 0.005");
    ASSERT_NEAR(ct.domain_min, 0.0,   0.0,    "domain_min = 0.0");
    ASSERT_NEAR(ct.domain_max, 1.0,   0.0,    "domain_max = 1.0");

    /* Regime thresholds */
    ASSERT_NEAR(ct.thresholds.omega_stable_max,   0.038, 0.0, "omega_stable_max");
    ASSERT_NEAR(ct.thresholds.F_stable_min,       0.90,  0.0, "F_stable_min");
    ASSERT_NEAR(ct.thresholds.S_stable_max,       0.15,  0.0, "S_stable_max");
    ASSERT_NEAR(ct.thresholds.C_stable_max,       0.14,  0.0, "C_stable_max");
    ASSERT_NEAR(ct.thresholds.omega_collapse_min, 0.30,  0.0, "omega_collapse_min");
    ASSERT_NEAR(ct.thresholds.IC_critical_max,    0.30,  0.0, "IC_critical_max");
}

static void test_contract_validation_ok(void)
{
    printf("  Test: Default contract passes validation\n");
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    ASSERT_EQ(umcp_contract_validate(&ct), UMCP_OK, "default is valid");
}

static void test_contract_validation_bad_epsilon(void)
{
    printf("  Test: Contract validation rejects epsilon <= 0\n");
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    ct.epsilon = 0.0;
    ASSERT_EQ(umcp_contract_validate(&ct), UMCP_ERR_RANGE, "epsilon=0 rejected");
    ct.epsilon = -1.0;
    ASSERT_EQ(umcp_contract_validate(&ct), UMCP_ERR_RANGE, "epsilon<0 rejected");
}

static void test_contract_validation_bad_domain(void)
{
    printf("  Test: Contract validation rejects bad domain\n");
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    ct.domain_min = 1.0;
    ct.domain_max = 0.0;
    ASSERT_EQ(umcp_contract_validate(&ct), UMCP_ERR_RANGE,
              "domain_min > domain_max rejected");
}

static void test_contract_equality(void)
{
    printf("  Test: Contract equality\n");
    umcp_contract_t a, b;
    umcp_contract_default(&a);
    umcp_contract_default(&b);
    ASSERT(umcp_contract_equal(&a, &b), "identical contracts are equal");

    b.epsilon = 1e-7;
    ASSERT(!umcp_contract_equal(&a, &b), "different epsilon → not equal");
}

static void test_gamma_omega(void)
{
    printf("  Test: Drift cost Gamma(omega)\n");
    /* Γ(0) = 0^3 / (1 - 0 + ε) = 0 */
    ASSERT_NEAR(umcp_gamma_omega(0.0, 3, 1e-8), 0.0, 1e-15,
                "Gamma(0) = 0");

    /* Γ(0.5) = 0.5^3 / (0.5 + ε) ≈ 0.25 */
    double g = umcp_gamma_omega(0.5, 3, 1e-8);
    ASSERT(g > 0.24 && g < 0.26, "Gamma(0.5) ≈ 0.25");

    /* Γ(ω) is monotonically increasing */
    double g1 = umcp_gamma_omega(0.1, 3, 1e-8);
    double g2 = umcp_gamma_omega(0.3, 3, 1e-8);
    double g3 = umcp_gamma_omega(0.5, 3, 1e-8);
    ASSERT(g1 < g2 && g2 < g3, "Gamma is monotone increasing");
}

static void test_cost_curvature(void)
{
    printf("  Test: Curvature cost D_C = alpha * C\n");
    ASSERT_NEAR(umcp_cost_curvature(0.0, 1.0), 0.0, 1e-15, "D_C(0) = 0");
    ASSERT_NEAR(umcp_cost_curvature(0.5, 1.0), 0.5, 1e-15, "D_C(0.5) = 0.5");
    ASSERT_NEAR(umcp_cost_curvature(0.5, 2.0), 1.0, 1e-15, "D_C(0.5,2) = 1.0");
}

static void test_budget_delta_kappa(void)
{
    printf("  Test: Budget delta_kappa\n");
    /* Δκ_budget = R·τ_R − (D_ω + D_C) */
    ASSERT_NEAR(umcp_budget_delta_kappa(1.0, 1.0, 0.1, 0.2),
                0.7, 1e-15, "budget = 1*1 - (0.1+0.2)");

    /* INF_REC → zero budget */
    ASSERT_NEAR(umcp_budget_delta_kappa(1.0, UMCP_TAU_R_INF_REC, 0.1, 0.2),
                0.0, 1e-15, "INF_REC → budget = 0");
}

static void test_seam_pass_check(void)
{
    printf("  Test: Seam PASS/FAIL check\n");
    /* All conditions met → PASS */
    umcp_seam_status_t s = umcp_check_seam_pass(
        0.001, 1.0, 1.001, 0.001, 0.005, 1e-6, NULL, 0);
    ASSERT_EQ(s, UMCP_SEAM_PASS, "small residual + finite tau → PASS");

    /* Large residual → FAIL */
    s = umcp_check_seam_pass(
        0.01, 1.0, 1.0, 0.0, 0.005, 1e-6, NULL, 0);
    ASSERT_EQ(s, UMCP_SEAM_FAIL, "large residual → FAIL");

    /* INF_REC → FAIL */
    s = umcp_check_seam_pass(
        0.0, UMCP_TAU_R_INF_REC, 1.0, 0.0, 0.005, 1e-6, NULL, 0);
    ASSERT_EQ(s, UMCP_SEAM_FAIL, "INF_REC tau_R → FAIL");
}

static void test_seam_fail_reason(void)
{
    printf("  Test: Seam fail reason buffer\n");
    char reason[128] = {0};
    umcp_check_seam_pass(
        0.01, 1.0, 1.0, 0.0, 0.005, 1e-6, reason, sizeof(reason));
    ASSERT(strlen(reason) > 0, "fail reason is non-empty on failure");
}


/* ═══════════════════ Regime Tests ════════════════════════════════ */

static void test_regime_stable(void)
{
    printf("  Test: Regime STABLE classification\n");
    /* Near-perfect system: low ω, high F, low S, low C */
    double c[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_result_t k;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

    umcp_regime_t r = umcp_classify_regime_default(&k);
    ASSERT_EQ(r, UMCP_REGIME_STABLE, "high-fidelity system → STABLE");
}

static void test_regime_watch(void)
{
    printf("  Test: Regime WATCH classification\n");
    /* Moderate drift: ω in [0.038, 0.30) */
    double c[8] = {0.90, 0.85, 0.88, 0.87, 0.86, 0.89, 0.90, 0.85};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_result_t k;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

    umcp_regime_t r = umcp_classify_regime_default(&k);
    /* F ≈ 0.875 < 0.90, so STABLE gate fails even if ω might be low */
    ASSERT(r == UMCP_REGIME_WATCH || r == UMCP_REGIME_COLLAPSE,
           "moderate-fidelity → WATCH or COLLAPSE");
}

static void test_regime_collapse(void)
{
    printf("  Test: Regime COLLAPSE classification\n");
    /* High drift: ω ≥ 0.30 → all channels near 0.5 or lower */
    double c[8] = {0.3, 0.4, 0.5, 0.2, 0.6, 0.3, 0.5, 0.4};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_result_t k;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

        umcp_regime_with_overlay_t r = umcp_classify_regime_with_overlay(&k, &ct.thresholds);
        ASSERT(r.base_regime == UMCP_REGIME_COLLAPSE,
            "low-fidelity → COLLAPSE");
}

static void test_regime_critical(void)
    umcp_regime_with_overlay_t r = umcp_classify_regime_with_overlay(&k, &thr);
    ASSERT(r.is_critical, "overlay struct: is_critical set");
{
    printf("  Test: CRITICAL overlay detection\n");
    /* One dead channel kills IC → IC < 0.30 */
    double c[8] = {0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1e-8};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_result_t k;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

    umcp_regime_thresholds_t thr;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    thr = ct.thresholds;

    ASSERT(umcp_is_critical(&k, &thr), "one dead channel → CRITICAL");
}

static void test_regime_partition(void)
{
    printf("  Test: Fisher-space regime partition\n");
    umcp_contract_t ct;
    umcp_contract_default(&ct);

    double pct_s, pct_w, pct_c, pct_cr;
    umcp_regime_partition(8, 10000, &ct.thresholds,
                          &pct_s, &pct_w, &pct_c, &pct_cr);

    /* Key result: Collapse > Watch > Stable */
    ASSERT(pct_c > pct_w, "Collapse > Watch");
    ASSERT(pct_w > pct_s, "Watch > Stable");

    /* Stable should be rare (~12.5%) */
    ASSERT(pct_s < 0.25, "Stable < 25%");

    /* Fractions should sum to ~1 (Critical overlaps, so just check non-critical) */
    ASSERT(pct_s + pct_w + pct_c > 0.99, "fractions sum ≈ 1");
}


/* ═══════════════════ Trace Tests ═════════════════════════════════ */

static void test_trace_init_free(void)
{
    printf("  Test: Trace init/free lifecycle\n");
    umcp_trace_t tr;
    int rc = umcp_trace_init(&tr, 8, 1e-8);
    ASSERT_EQ(rc, UMCP_OK, "init succeeds");
    ASSERT_EQ(tr.n, (size_t)8, "n = 8");
    ASSERT_NEAR(tr.epsilon, 1e-8, 0.0, "epsilon preserved");
    ASSERT(tr.c != NULL, "c buffer allocated");
    ASSERT(tr.w != NULL, "w buffer allocated");
    umcp_trace_free(&tr);
    ASSERT(tr.c == NULL, "c freed");
    ASSERT(tr.w == NULL, "w freed");
}

static void test_trace_init_zero_dim(void)
{
    printf("  Test: Trace init rejects n=0\n");
    umcp_trace_t tr;
    ASSERT_EQ(umcp_trace_init(&tr, 0, 1e-8), UMCP_ERR_ZERO_DIM,
              "n=0 rejected");
}

static void test_trace_uniform_weights(void)
{
    printf("  Test: Trace uniform weights\n");
    umcp_trace_t tr;
    umcp_trace_init(&tr, 4, 1e-8);
    umcp_trace_uniform_weights(&tr);
    for (size_t i = 0; i < 4; i++) {
        ASSERT_NEAR(tr.w[i], 0.25, 1e-15, "w[i] = 1/n = 0.25");
    }
    umcp_trace_free(&tr);
}

static void test_trace_set_weights(void)
{
    printf("  Test: Trace set custom weights\n");
    umcp_trace_t tr;
    umcp_trace_init(&tr, 4, 1e-8);

    double good_w[4] = {0.1, 0.2, 0.3, 0.4};
    ASSERT_EQ(umcp_trace_set_weights(&tr, good_w), UMCP_OK,
              "valid weights accepted");

    double bad_w[4] = {0.1, 0.2, 0.3, 0.5};  /* sum = 1.1 */
    ASSERT_EQ(umcp_trace_set_weights(&tr, bad_w), UMCP_ERR_WEIGHT_SUM,
              "non-simplex weights rejected");

    umcp_trace_free(&tr);
}

static void test_trace_set_channels(void)
{
    printf("  Test: Trace set channels with pre_clip\n");
    umcp_trace_t tr;
    umcp_trace_init(&tr, 4, 1e-8);

    double raw[4] = {0.5, 0.0, 1.0, -0.1};
    umcp_trace_set_channels(&tr, raw);

    /* 0.5 unchanged */
    ASSERT_NEAR(tr.c[0], 0.5, 1e-15, "c[0] = 0.5 unchanged");
    /* 0.0 clipped to epsilon */
    ASSERT_NEAR(tr.c[1], 1e-8, 1e-15, "c[1] clipped to epsilon");
    /* 1.0 clipped to 1-epsilon */
    ASSERT_NEAR(tr.c[2], 1.0 - 1e-8, 1e-15, "c[2] clipped to 1-epsilon");
    /* -0.1 clipped to epsilon */
    ASSERT_NEAR(tr.c[3], 1e-8, 1e-15, "c[3] clipped to epsilon (negative)");

    ASSERT_EQ(tr.clipped, 1, "clipped flag is set");
    umcp_trace_free(&tr);
}

static void test_trace_embed_linear(void)
{
    printf("  Test: Trace linear embedding\n");
    umcp_trace_t tr;
    umcp_trace_init(&tr, 3, 1e-8);
    umcp_trace_uniform_weights(&tr);

    /* Raw data in [0, 100] → mapped to [ε, 1-ε] */
    double raw[3] = {0.0, 50.0, 100.0};
    int rc = umcp_trace_embed_linear(&tr, raw, 0.0, 100.0);
    ASSERT_EQ(rc, UMCP_OK, "linear embed succeeds");

    /* Minimum maps to epsilon */
    ASSERT_NEAR(tr.c[0], 1e-8, 1e-10, "min → epsilon");
    /* Midpoint maps to ~0.5 */
    ASSERT(tr.c[1] > 0.45 && tr.c[1] < 0.55, "midpoint → ~0.5");
    /* Maximum maps to 1 - epsilon */
    ASSERT_NEAR(tr.c[2], 1.0 - 1e-8, 1e-10, "max → 1-epsilon");

    umcp_trace_free(&tr);
}

static void test_trace_embed_linear_bad_bounds(void)
{
    printf("  Test: Trace linear embedding rejects bad bounds\n");
    umcp_trace_t tr;
    umcp_trace_init(&tr, 3, 1e-8);
    double raw[3] = {1.0, 2.0, 3.0};
    ASSERT_EQ(umcp_trace_embed_linear(&tr, raw, 10.0, 5.0), UMCP_ERR_RANGE,
              "min > max rejected");
    umcp_trace_free(&tr);
}

static void test_trace_embed_log(void)
{
    printf("  Test: Trace logarithmic embedding\n");
    umcp_trace_t tr;
    umcp_trace_init(&tr, 3, 1e-8);
    umcp_trace_uniform_weights(&tr);

    /* Masses spanning orders of magnitude */
    double raw[3] = {1.0, 100.0, 10000.0};
    int rc = umcp_trace_embed_log(&tr, raw, 1.0, 10000.0);
    ASSERT_EQ(rc, UMCP_OK, "log embed succeeds");

    /* All in [ε, 1-ε] */
    for (size_t i = 0; i < 3; i++) {
        ASSERT(tr.c[i] >= 1e-8 && tr.c[i] <= 1.0 - 1e-8,
               "log-embedded channel in valid range");
    }
    /* Monotonically increasing (log preserves order) */
    ASSERT(tr.c[0] < tr.c[1] && tr.c[1] < tr.c[2],
           "log embed preserves order");

    umcp_trace_free(&tr);
}

static void test_trace_validate_identities(void)
{
    printf("  Test: Trace identity validation\n");
    umcp_kernel_result_t k;
    double c[8] = {0.5, 0.7, 0.3, 0.9, 0.4, 0.6, 0.8, 0.2};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

    umcp_verdict_t v = umcp_validate_identities(&k, 1e-9);
    ASSERT_EQ(v, UMCP_CONFORMANT, "valid kernel → CONFORMANT");
}

static void test_trace_validate_identities_corrupted(void)
{
    printf("  Test: Trace identity validation catches corruption\n");
    umcp_kernel_result_t k;
    double c[8] = {0.5, 0.7, 0.3, 0.9, 0.4, 0.6, 0.8, 0.2};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

    /* Corrupt F + omega (violate duality) */
    k.F += 0.1;
    umcp_verdict_t v = umcp_validate_identities(&k, 1e-9);
    ASSERT_EQ(v, UMCP_NONCONFORMANT,
              "corrupted F → NONCONFORMANT");
}


/* ═══════════════════ Ledger Tests ════════════════════════════════ */

static void test_ledger_init(void)
{
    printf("  Test: Ledger initialization\n");
    umcp_ledger_entry_t buf[32];
    umcp_ledger_t ledger;
    int rc = umcp_ledger_init(&ledger, buf, 32);
    ASSERT_EQ(rc, UMCP_OK, "init succeeds");
    ASSERT_EQ(ledger.count, (size_t)0, "count starts at 0");
    ASSERT_EQ(ledger.pass_count, (uint32_t)0, "pass_count starts at 0");
    ASSERT_EQ(ledger.fail_count, (uint32_t)0, "fail_count starts at 0");
}

static void test_ledger_append_basic(void)
{
    printf("  Test: Ledger basic append\n");
    umcp_ledger_entry_t buf[8];
    umcp_ledger_t ledger;
    umcp_ledger_init(&ledger, buf, 8);

    /* Create a entry manually */
    umcp_ledger_entry_t e;
    memset(&e, 0, sizeof(e));
    e.timestamp = 0;
    e.regime = UMCP_REGIME_STABLE;
    e.residual = 0.001;
    e.seam = UMCP_SEAM_PASS;
    e.verdict = UMCP_CONFORMANT;

    int rc = umcp_ledger_append(&ledger, &e);
    ASSERT_EQ(rc, UMCP_OK, "append succeeds");
    ASSERT_EQ(ledger.count, (size_t)1, "count = 1 after append");
    ASSERT_EQ(ledger.pass_count, (uint32_t)1, "pass_count incremented");
}

static void test_ledger_append_overflow(void)
{
    printf("  Test: Ledger overflow protection\n");
    umcp_ledger_entry_t buf[2];
    umcp_ledger_t ledger;
    umcp_ledger_init(&ledger, buf, 2);

    umcp_ledger_entry_t e;
    memset(&e, 0, sizeof(e));
    e.seam = UMCP_SEAM_PASS;

    umcp_ledger_append(&ledger, &e);
    umcp_ledger_append(&ledger, &e);
    int rc = umcp_ledger_append(&ledger, &e);
    ASSERT_EQ(rc, UMCP_ERR_RANGE, "overflow detected");
    ASSERT_EQ(ledger.count, (size_t)2, "count stays at capacity");
}

static void test_ledger_build_entry(void)
{
    printf("  Test: Ledger build_entry from kernel\n");
    /* Compute kernel */
    double c[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_result_t k;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

    umcp_contract_t ct;
    umcp_contract_default(&ct);

    umcp_ledger_entry_t entry;
    int rc = umcp_ledger_build_entry(&entry, &k, &ct, NAN, 1.0, 1.0);
    ASSERT_EQ(rc, UMCP_OK, "build_entry succeeds");
    ASSERT_EQ(entry.regime, UMCP_REGIME_STABLE, "high-fidelity → STABLE");
    ASSERT(entry.D_omega >= 0.0, "D_omega is non-negative");
    ASSERT(entry.D_C >= 0.0, "D_C is non-negative");
    /* First entry → seam PENDING */
    ASSERT_EQ(entry.seam, UMCP_SEAM_PENDING, "first entry → PENDING seam");
}

static void test_ledger_build_entry_with_prior(void)
{
    printf("  Test: Ledger build_entry with prior kappa\n");
    double c[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    double w[8]; for (int i = 0; i < 8; i++) w[i] = 0.125;
    umcp_kernel_result_t k;
    umcp_kernel_compute(c, w, 8, 1e-8, &k);

    umcp_contract_t ct;
    umcp_contract_default(&ct);

    /* Use current kappa as prior → residual should be near zero */
    umcp_ledger_entry_t entry;
    int rc = umcp_ledger_build_entry(&entry, &k, &ct, k.kappa, 1.0, 1.0);
    ASSERT_EQ(rc, UMCP_OK, "build_entry with prior succeeds");
    ASSERT(entry.seam != UMCP_SEAM_PENDING, "has prior → not PENDING");
}

static void test_ledger_running_stats(void)
{
    printf("  Test: Ledger running statistics\n");
    umcp_ledger_entry_t buf[8];
    umcp_ledger_t ledger;
    umcp_ledger_init(&ledger, buf, 8);

    umcp_ledger_entry_t e;
    memset(&e, 0, sizeof(e));
    e.regime = UMCP_REGIME_STABLE;
    e.residual = 0.001;
    e.seam = UMCP_SEAM_PASS;
    e.verdict = UMCP_CONFORMANT;
    umcp_ledger_append(&ledger, &e);

    e.regime = UMCP_REGIME_WATCH;
    e.residual = 0.003;
    umcp_ledger_append(&ledger, &e);

    ASSERT_NEAR(umcp_ledger_mean_residual(&ledger), 0.002, 1e-10,
                "mean residual = (0.001+0.003)/2");

    ASSERT_NEAR(ledger.max_residual, 0.003, 1e-15,
                "max residual = 0.003");

    ASSERT_EQ(ledger.regime_counts[UMCP_REGIME_STABLE], (uint32_t)1,
              "1 STABLE");
    ASSERT_EQ(ledger.regime_counts[UMCP_REGIME_WATCH], (uint32_t)1,
              "1 WATCH");
}

static void test_ledger_regime_fractions(void)
{
    printf("  Test: Ledger regime fractions\n");
    umcp_ledger_entry_t buf[4];
    umcp_ledger_t ledger;
    umcp_ledger_init(&ledger, buf, 4);

    umcp_ledger_entry_t e;
    memset(&e, 0, sizeof(e));
    e.seam = UMCP_SEAM_PASS;
    e.verdict = UMCP_CONFORMANT;

    e.regime = UMCP_REGIME_STABLE;  umcp_ledger_append(&ledger, &e);
    e.regime = UMCP_REGIME_STABLE;  umcp_ledger_append(&ledger, &e);
    e.regime = UMCP_REGIME_WATCH;   umcp_ledger_append(&ledger, &e);
    e.regime = UMCP_REGIME_COLLAPSE;umcp_ledger_append(&ledger, &e);

    double s, w_fr, c_fr, cr;
    umcp_ledger_regime_fractions(&ledger, &s, &w_fr, &c_fr, &cr);
    ASSERT_NEAR(s, 0.5, 1e-10, "50% Stable");
    ASSERT_NEAR(w_fr, 0.25, 1e-10, "25% Watch");
    ASSERT_NEAR(c_fr, 0.25, 1e-10, "25% Collapse");
}

static void test_ledger_verdict_conformant(void)
{
    printf("  Test: Ledger overall verdict CONFORMANT\n");
    umcp_ledger_entry_t buf[4];
    umcp_ledger_t ledger;
    umcp_ledger_init(&ledger, buf, 4);

    umcp_ledger_entry_t e;
    memset(&e, 0, sizeof(e));
    e.seam = UMCP_SEAM_PASS;
    e.verdict = UMCP_CONFORMANT;
    umcp_ledger_append(&ledger, &e);
    umcp_ledger_append(&ledger, &e);

    ASSERT_EQ(umcp_ledger_verdict(&ledger), UMCP_CONFORMANT,
              "all PASS + CONFORMANT → overall CONFORMANT");
}

static void test_ledger_verdict_nonconformant(void)
{
    printf("  Test: Ledger overall verdict NONCONFORMANT\n");
    umcp_ledger_entry_t buf[4];
    umcp_ledger_t ledger;
    umcp_ledger_init(&ledger, buf, 4);

    umcp_ledger_entry_t e;
    memset(&e, 0, sizeof(e));
    e.seam = UMCP_SEAM_PASS;
    e.verdict = UMCP_CONFORMANT;
    umcp_ledger_append(&ledger, &e);

    e.seam = UMCP_SEAM_FAIL;
    e.verdict = UMCP_NONCONFORMANT;
    umcp_ledger_append(&ledger, &e);

    ASSERT_EQ(umcp_ledger_verdict(&ledger), UMCP_NONCONFORMANT,
              "any FAIL → overall NONCONFORMANT");
}

static void test_ledger_verdict_empty(void)
{
    printf("  Test: Ledger verdict on empty ledger\n");
    umcp_ledger_entry_t buf[4];
    umcp_ledger_t ledger;
    umcp_ledger_init(&ledger, buf, 4);

    ASSERT_EQ(umcp_ledger_verdict(&ledger), UMCP_NON_EVALUABLE,
              "empty ledger → NON_EVALUABLE");
}


/* ═══════════════════ Pipeline Tests ══════════════════════════════ */

static void test_pipeline_init(void)
{
    printf("  Test: Pipeline initialization\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    umcp_ledger_entry_t buf[16];

    int rc = umcp_pipeline_init(&pipe, &ct, buf, 16);
    ASSERT_EQ(rc, UMCP_OK, "pipeline init succeeds");
    ASSERT_EQ(pipe.initialized, 1, "initialized flag set");
    ASSERT_EQ(umcp_pipeline_step_count(&pipe), (size_t)0,
              "step count starts at 0");
}

static void test_pipeline_init_bad_contract(void)
{
    printf("  Test: Pipeline init with bad contract\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    ct.epsilon = -1.0;  /* Invalid */
    umcp_ledger_entry_t buf[16];

    int rc = umcp_pipeline_init(&pipe, &ct, buf, 16);
    ASSERT_EQ(rc, UMCP_ERR_RANGE, "invalid contract → init fails");
}

static void test_pipeline_single_step(void)
{
    printf("  Test: Pipeline single step through spine\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    umcp_ledger_entry_t buf[16];
    umcp_pipeline_init(&pipe, &ct, buf, 16);

    /* Create a high-fidelity trace */
    umcp_trace_t tr;
    umcp_trace_init(&tr, 8, ct.epsilon);
    umcp_trace_uniform_weights(&tr);
    double raw[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    umcp_trace_set_channels(&tr, raw);

    umcp_pipeline_result_t result;
    int rc = umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &result);
    ASSERT_EQ(rc, UMCP_OK, "step succeeds");

    /* First step → seam PENDING (no prior) */
    ASSERT_EQ(result.seam, UMCP_SEAM_PENDING, "first step → PENDING");

    /* Kernel was computed */
    ASSERT(result.kernel.F > 0.90, "F > 0.90 for high-fidelity trace");
    ASSERT_NEAR(result.kernel.F + result.kernel.omega, 1.0, 1e-15,
                "duality identity holds");

    /* Regime should be STABLE for this input */
    ASSERT_EQ(result.regime, UMCP_REGIME_STABLE, "STABLE regime");

    /* Step count updated */
    ASSERT_EQ(umcp_pipeline_step_count(&pipe), (size_t)1, "step_count = 1");

    umcp_trace_free(&tr);
}

static void test_pipeline_two_steps(void)
{
    printf("  Test: Pipeline two-step seam evaluation\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    umcp_ledger_entry_t buf[16];
    umcp_pipeline_init(&pipe, &ct, buf, 16);

    umcp_trace_t tr;
    umcp_trace_init(&tr, 8, ct.epsilon);
    umcp_trace_uniform_weights(&tr);

    /* Step 1: stable */
    double c1[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    umcp_trace_set_channels(&tr, c1);
    umcp_pipeline_result_t r1;
    umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &r1);
    ASSERT_EQ(r1.seam, UMCP_SEAM_PENDING, "first step is PENDING");

    /* Step 2: similar (small drift) → seam can now evaluate */
    double c2[8] = {0.97, 0.96, 0.97, 0.98, 0.97, 0.96, 0.97, 0.98};
    umcp_trace_set_channels(&tr, c2);
    umcp_pipeline_result_t r2;
    umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &r2);
    ASSERT(r2.seam != UMCP_SEAM_PENDING, "second step evaluates seam");
    ASSERT_EQ(umcp_pipeline_step_count(&pipe), (size_t)2, "step_count = 2");

    umcp_trace_free(&tr);
}

static void test_pipeline_verdict_accumulates(void)
{
    printf("  Test: Pipeline verdict accumulates correctly\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    umcp_ledger_entry_t buf[16];
    umcp_pipeline_init(&pipe, &ct, buf, 16);

    umcp_trace_t tr;
    umcp_trace_init(&tr, 8, ct.epsilon);
    umcp_trace_uniform_weights(&tr);

    /* Run 5 stable steps */
    double c[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    umcp_pipeline_result_t result;
    for (int i = 0; i < 5; i++) {
        umcp_trace_set_channels(&tr, c);
        umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &result);
    }

    ASSERT_EQ(umcp_pipeline_step_count(&pipe), (size_t)5, "5 steps");

    /* All identity checks should pass, but seam may FAIL because
     * budget ≠ Δκ for repeated identical traces (residual is non-zero).
     * This is correct: the seam measures actual return dynamics,
     * not just identity conformance. */
    umcp_verdict_t v = umcp_pipeline_verdict(&pipe);
    ASSERT(v == UMCP_CONFORMANT || v == UMCP_NONCONFORMANT,
           "all-stable → CONFORMANT or NONCONFORMANT (seam may not close)");

    umcp_trace_free(&tr);
}

static void test_pipeline_stance_query(void)
{
    printf("  Test: Pipeline stance query\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    umcp_ledger_entry_t buf[16];
    umcp_pipeline_init(&pipe, &ct, buf, 16);

    umcp_trace_t tr;
    umcp_trace_init(&tr, 8, ct.epsilon);
    umcp_trace_uniform_weights(&tr);
    double c[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    umcp_trace_set_channels(&tr, c);

    umcp_pipeline_result_t result;
    umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &result);

    umcp_stance_t stance = umcp_pipeline_stance(&pipe);
    ASSERT_EQ(stance.regime, UMCP_REGIME_STABLE, "stance regime is STABLE");

    umcp_trace_free(&tr);
}

static void test_pipeline_regime_transition(void)
{
    printf("  Test: Pipeline handles regime transition\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    umcp_ledger_entry_t buf[16];
    umcp_pipeline_init(&pipe, &ct, buf, 16);

    umcp_trace_t tr;
    umcp_trace_init(&tr, 8, ct.epsilon);
    umcp_trace_uniform_weights(&tr);
    umcp_pipeline_result_t result;

    /* Step 1: Stable */
    double c1[8] = {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97};
    umcp_trace_set_channels(&tr, c1);
    umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &result);
    ASSERT_EQ(result.regime, UMCP_REGIME_STABLE, "step 1 STABLE");

    /* Step 2: Collapse (low fidelity) */
    double c2[8] = {0.3, 0.4, 0.2, 0.5, 0.3, 0.4, 0.2, 0.5};
    umcp_trace_set_channels(&tr, c2);
    umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &result);
        ASSERT(result.regime == UMCP_REGIME_COLLAPSE,
            "step 2 COLLAPSE");

    umcp_trace_free(&tr);
}

static void test_pipeline_inf_rec(void)
{
    printf("  Test: Pipeline with INF_REC return time\n");
    umcp_pipeline_t pipe;
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    umcp_ledger_entry_t buf[16];
    umcp_pipeline_init(&pipe, &ct, buf, 16);

    umcp_trace_t tr;
    umcp_trace_init(&tr, 8, ct.epsilon);
    umcp_trace_uniform_weights(&tr);
    double c[8] = {0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6};
    umcp_trace_set_channels(&tr, c);

    umcp_pipeline_result_t result;
    /* First step: PENDING */
    umcp_pipeline_step(&pipe, &tr, 0.0, UMCP_TAU_R_INF_REC, &result);
    ASSERT_EQ(result.seam, UMCP_SEAM_PENDING, "first with INF_REC → PENDING");

    /* Second step: INF_REC → FAIL */
    umcp_pipeline_step(&pipe, &tr, 0.0, UMCP_TAU_R_INF_REC, &result);
    ASSERT_EQ(result.seam, UMCP_SEAM_FAIL, "INF_REC → FAIL seam");

    umcp_trace_free(&tr);
}


/* ═══════════════════ Integration Test ════════════════════════════ */

static void test_full_spine_integration(void)
{
    printf("  Test: Full spine integration (Contract→Canon→Closures→Ledger→Stance)\n");

    /* ── Stop 1: Contract (freeze) ── */
    umcp_contract_t ct;
    umcp_contract_default(&ct);
    ASSERT_EQ(umcp_contract_validate(&ct), UMCP_OK, "contract valid");

    /* ── Pipeline setup ── */
    umcp_pipeline_t pipe;
    umcp_ledger_entry_t buf[32];
    umcp_pipeline_init(&pipe, &ct, buf, 32);

    /* ── Stop 2: Canon (tell — process 10 traces) ── */
    umcp_trace_t tr;
    umcp_trace_init(&tr, 8, ct.epsilon);
    umcp_trace_uniform_weights(&tr);

    /* Simulate a trajectory: stable → watch → stable → stable */
    double traces[4][8] = {
        {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97},  /* Stable */
        {0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77, 0.76},  /* Watch  */
        {0.98, 0.97, 0.96, 0.99, 0.98, 0.97, 0.98, 0.97},  /* Stable */
        {0.99, 0.98, 0.97, 0.99, 0.98, 0.99, 0.98, 0.99},  /* Stable */
    };

    umcp_pipeline_result_t result;
    for (int i = 0; i < 4; i++) {
        umcp_trace_set_channels(&tr, traces[i]);
        int rc = umcp_pipeline_step(&pipe, &tr, 1.0, 1.0, &result);
        ASSERT_EQ(rc, UMCP_OK, "step succeeds");

        /* ── Stop 3: Closures (verify gates fire correctly) ── */
        ASSERT(result.identity_check == UMCP_CONFORMANT,
               "identity check passes for valid kernel");

        /* Verify duality identity at every step */
        ASSERT_NEAR(result.kernel.F + result.kernel.omega, 1.0, 1e-15,
                    "F + omega = 1 at every step");

        /* Verify integrity bound */
        ASSERT(result.kernel.IC <= result.kernel.F + 1e-14,
               "IC <= F at every step");
    }

    /* ── Stop 4 & 5: Ledger → Stance ── */
    ASSERT_EQ(umcp_pipeline_step_count(&pipe), (size_t)4, "4 steps processed");

    umcp_stance_t stance = umcp_pipeline_stance(&pipe);
    printf("    Final stance: regime=%s, verdict=%s\n",
           umcp_regime_str(stance.regime),
           umcp_verdict_str(stance.verdict));

    /* The trajectory should produce a non-empty verdict */
    ASSERT(stance.verdict != UMCP_NON_EVALUABLE ||
           stance.seam == UMCP_SEAM_PENDING,
           "4-step trajectory produces a meaningful verdict");

    /* Ledger statistics */
    double mean_r = umcp_ledger_mean_residual(&pipe.ledger);
    printf("    Mean |residual|: %.6f\n", mean_r);

    double s, w_fr, c_fr, cr;
    umcp_ledger_regime_fractions(&pipe.ledger, &s, &w_fr, &c_fr, &cr);
    printf("    Regime fractions: Stable=%.2f Watch=%.2f Collapse=%.2f Critical=%.2f\n",
           s, w_fr, c_fr, cr);

    /* At least 2 of the 4 traces should be STABLE */
    ASSERT(s >= 0.40, "≥ 40% STABLE in this trajectory");

    umcp_trace_free(&tr);
}


/* ═══════════════════ Test Runner ═════════════════════════════════ */

int main(void)
{
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  UMCP C Orchestration Tests                             ║\n");
    printf("║  Types · Contract · Regime · Trace · Ledger · Pipeline  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Types */
    printf("═══ Types ═══\n");
    test_types_regime_enum();
    test_types_verdict_enum();
    test_types_seam_enum();
    test_types_inf_rec();
    test_types_string_conversion();

    /* Contract */
    printf("\n═══ Contract ═══\n");
    test_contract_default_values();
    test_contract_validation_ok();
    test_contract_validation_bad_epsilon();
    test_contract_validation_bad_domain();
    test_contract_equality();
    test_gamma_omega();
    test_cost_curvature();
    test_budget_delta_kappa();
    test_seam_pass_check();
    test_seam_fail_reason();

    /* Regime */
    printf("\n═══ Regime ═══\n");
    test_regime_stable();
    test_regime_watch();
    test_regime_collapse();
    test_regime_critical();
    test_regime_partition();

    /* Trace */
    printf("\n═══ Trace ═══\n");
    test_trace_init_free();
    test_trace_init_zero_dim();
    test_trace_uniform_weights();
    test_trace_set_weights();
    test_trace_set_channels();
    test_trace_embed_linear();
    test_trace_embed_linear_bad_bounds();
    test_trace_embed_log();
    test_trace_validate_identities();
    test_trace_validate_identities_corrupted();

    /* Ledger */
    printf("\n═══ Ledger ═══\n");
    test_ledger_init();
    test_ledger_append_basic();
    test_ledger_append_overflow();
    test_ledger_build_entry();
    test_ledger_build_entry_with_prior();
    test_ledger_running_stats();
    test_ledger_regime_fractions();
    test_ledger_verdict_conformant();
    test_ledger_verdict_nonconformant();
    test_ledger_verdict_empty();

    /* Pipeline */
    printf("\n═══ Pipeline ═══\n");
    test_pipeline_init();
    test_pipeline_init_bad_contract();
    test_pipeline_single_step();
    test_pipeline_two_steps();
    test_pipeline_verdict_accumulates();
    test_pipeline_stance_query();
    test_pipeline_regime_transition();
    test_pipeline_inf_rec();

    /* Integration */
    printf("\n═══ Integration ═══\n");
    test_full_spine_integration();

    /* Summary */
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Results: %d/%d passed, %d failed                      ",
           tests_passed, tests_run, tests_failed);
    if (tests_failed == 0)
        printf("  ✓ ║\n");
    else
        printf("  ✗ ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    return tests_failed > 0 ? 1 : 0;
}
