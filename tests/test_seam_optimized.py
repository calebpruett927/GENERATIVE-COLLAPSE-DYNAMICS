"""
Tests for seam_optimized.py - Seam chain accounting.

These tests verify the seam accounting implementation against
the formal specifications in KERNEL_SPECIFICATION.md (Lemmas 18-21, 27).
"""

from umcp.seam_optimized import (
    SeamChainAccumulator,
    SeamChainMetrics,
    SeamRecord,
)


class TestSeamRecord:
    """Tests for the SeamRecord dataclass."""

    def test_record_creation(self):
        """SeamRecord can be created with required fields."""
        record = SeamRecord(
            t0=0,
            t1=10,
            kappa_t0=-0.1,
            kappa_t1=-0.05,
            tau_R=10.0,
            delta_kappa_ledger=0.05,
            delta_kappa_budget=0.06,
            residual=0.01,
            cumulative_residual=0.01,
        )
        assert record.t0 == 0
        assert record.t1 == 10
        assert record.tau_R == 10.0

    def test_residual_computation(self):
        """Residual is budget - ledger."""
        record = SeamRecord(
            t0=0,
            t1=10,
            kappa_t0=-0.1,
            kappa_t1=-0.05,
            tau_R=10.0,
            delta_kappa_ledger=0.05,
            delta_kappa_budget=0.06,
            residual=0.01,
            cumulative_residual=0.01,
        )
        assert abs(record.residual - (record.delta_kappa_budget - record.delta_kappa_ledger)) < 1e-10


class TestSeamChainAccumulator:
    """Tests for the SeamChainAccumulator class."""

    def test_accumulator_creation(self):
        """Accumulator can be created."""
        acc = SeamChainAccumulator()
        assert acc.total_delta_kappa == 0.0
        assert len(acc.seam_history) == 0

    def test_add_seam(self):
        """Seam can be added to accumulator."""
        acc = SeamChainAccumulator()
        record = acc.add_seam(t0=0, t1=10, kappa_t0=-0.1, kappa_t1=-0.05, tau_R=10.0)
        assert isinstance(record, SeamRecord)
        assert len(acc.seam_history) == 1

    def test_incremental_update(self):
        """Total delta_kappa updates incrementally (OPT-10)."""
        acc = SeamChainAccumulator()

        # Add first seam
        acc.add_seam(t0=0, t1=10, kappa_t0=-0.1, kappa_t1=-0.05, tau_R=10.0)
        first_total = acc.total_delta_kappa

        # Add second seam
        acc.add_seam(t0=10, t1=20, kappa_t0=-0.05, kappa_t1=-0.02, tau_R=10.0)
        second_total = acc.total_delta_kappa

        # Total should be sum of both
        assert second_total > first_total

    def test_residual_tracking(self):
        """Cumulative residual is tracked (OPT-11)."""
        acc = SeamChainAccumulator()

        acc.add_seam(t0=0, t1=10, kappa_t0=-0.1, kappa_t1=-0.05, tau_R=10.0)
        assert acc.cumulative_abs_residual >= 0

    def test_multiple_seams(self):
        """Multiple seams can be added."""
        acc = SeamChainAccumulator()

        for i in range(5):
            kappa_start = -0.1 + i * 0.01
            kappa_end = kappa_start + 0.01
            acc.add_seam(t0=i * 10, t1=(i + 1) * 10, kappa_t0=kappa_start, kappa_t1=kappa_end, tau_R=10.0)

        assert len(acc.seam_history) == 5

    def test_seam_chain_composition(self):
        """Lemma 20: delta_kappa_ledger composes additively."""
        acc = SeamChainAccumulator()

        # Chain of seams: t0->t1->t2
        record1 = acc.add_seam(t0=0, t1=10, kappa_t0=-0.3, kappa_t1=-0.2, tau_R=10.0)
        record2 = acc.add_seam(t0=10, t1=20, kappa_t0=-0.2, kappa_t1=-0.1, tau_R=10.0)

        # Total change should be sum of individual changes
        total_from_records = record1.delta_kappa_ledger + record2.delta_kappa_ledger
        assert abs(acc.total_delta_kappa - total_from_records) < 1e-10


class TestSeamChainMetrics:
    """Tests for the SeamChainMetrics dataclass."""

    def test_metrics_creation(self):
        """SeamChainMetrics can be created."""
        metrics = SeamChainMetrics(
            total_seams=5,
            total_delta_kappa=0.5,
            cumulative_abs_residual=0.02,
            max_residual=0.01,
            mean_residual=0.004,
            growth_exponent=0.5,
            is_returning=True,
            failure_detected=False,
        )
        assert metrics.total_seams == 5
        assert metrics.is_returning


class TestResidualBounds:
    """Tests for residual bound verification."""

    def test_sublinear_growth(self):
        """Returning systems have sublinear residual growth."""
        acc = SeamChainAccumulator()

        # Add seams with small, controlled residuals
        for i in range(20):
            kappa_start = -0.5 + i * 0.02
            kappa_end = kappa_start + 0.02
            acc.add_seam(
                t0=i * 10,
                t1=(i + 1) * 10,
                kappa_t0=kappa_start,
                kappa_t1=kappa_end,
                tau_R=10.0,
                R=0.01,  # Match budget to ledger closely
            )

        # For returning systems, residual should not grow linearly
        assert not acc.failure_detected
