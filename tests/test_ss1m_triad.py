"""Tests for SS1m triad checksum implementation.

Tests the mod-97 prime-field checksum system for edition identity.
"""

import pytest

from umcp.ss1m_triad import (
    PRIME_COEFFICIENTS,
    TRIAD_MODULUS,
    EditionCounts,
    EditionTriad,
    compute_triad,
    decode_base32,
    encode_base32,
    parse_triad,
    triad_to_eid12,
    verify_triad,
)


class TestTriadComputation:
    """Test basic triad computation."""

    def test_compute_triad_basic(self) -> None:
        """Test triad computation with simple inputs."""
        counts = EditionCounts(pages=10, figures=5, tables=2, equations=3, references=20)
        triad = compute_triad(counts)
        assert isinstance(triad, EditionTriad)
        assert 0 <= triad.c1 < TRIAD_MODULUS
        assert 0 <= triad.c2 < TRIAD_MODULUS
        assert 0 <= triad.c3 < TRIAD_MODULUS

    def test_compute_triad_deterministic(self) -> None:
        """Same inputs produce same triad."""
        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad1 = compute_triad(counts)
        triad2 = compute_triad(counts)
        assert triad1 == triad2

    def test_compute_triad_different_inputs(self) -> None:
        """Different inputs produce different triads."""
        counts1 = EditionCounts(pages=10, figures=5, tables=2, equations=3, references=20)
        counts2 = EditionCounts(pages=10, figures=6, tables=2, equations=3, references=20)  # Different figures
        triad1 = compute_triad(counts1)
        triad2 = compute_triad(counts2)
        # At least one component should differ
        assert (triad1.c1, triad1.c2, triad1.c3) != (triad2.c1, triad2.c2, triad2.c3)

    def test_compute_triad_modulus_range(self) -> None:
        """All components are in valid range for any inputs."""
        import random

        for _ in range(100):
            counts = EditionCounts(
                pages=random.randint(0, 1000),
                figures=random.randint(0, 1000),
                tables=random.randint(0, 100),
                equations=random.randint(0, 500),
                references=random.randint(0, 1000),
            )
            triad = compute_triad(counts)
            assert 0 <= triad.c1 < 97, f"c1={triad.c1} out of range"
            assert 0 <= triad.c2 < 97, f"c2={triad.c2} out of range"
            assert 0 <= triad.c3 < 97, f"c3={triad.c3} out of range"

    def test_c1_formula(self) -> None:
        """C1 = (P + F + T + E + R) mod 97 (canonical sum checksum)."""
        counts = EditionCounts(pages=10, figures=5, tables=3, equations=2, references=4)
        triad = compute_triad(counts)
        # Canonical: C1 = (P + F + T + E + R) mod 97
        assert triad.c1 == (10 + 5 + 3 + 2 + 4) % 97

    def test_c2_formula(self) -> None:
        """C2 = (1*P + 2*F + 3*T + 5*E + 7*R) mod 97."""
        counts = EditionCounts(pages=10, figures=5, tables=3, equations=2, references=4)
        triad = compute_triad(counts)
        expected_c2 = (1 * 10 + 2 * 5 + 3 * 3 + 5 * 2 + 7 * 4) % 97
        assert triad.c2 == expected_c2

    def test_c3_formula(self) -> None:
        """C3 = (P·F + T·E + R) mod 97 (canonical product checksum)."""
        counts = EditionCounts(pages=10, figures=8, tables=5, equations=3, references=7)
        triad = compute_triad(counts)
        # Canonical: C3 = (P·F + T·E + R) mod 97
        assert triad.c3 == (10 * 8 + 5 * 3 + 7) % 97


class TestTriadVerification:
    """Test triad verification."""

    def test_verify_valid_triad(self) -> None:
        """Valid triad passes verification."""
        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad = compute_triad(counts)
        assert verify_triad(counts, triad) is True

    def test_verify_invalid_triad_wrong_c1(self) -> None:
        """Tampered c1 fails verification."""
        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad = compute_triad(counts)
        tampered = EditionTriad(
            c1=(triad.c1 + 1) % 97,
            c2=triad.c2,
            c3=triad.c3,
        )
        assert verify_triad(counts, tampered) is False

    def test_verify_invalid_triad_wrong_c2(self) -> None:
        """Tampered c2 fails verification."""
        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad = compute_triad(counts)
        tampered = EditionTriad(
            c1=triad.c1,
            c2=(triad.c2 + 1) % 97,
            c3=triad.c3,
        )
        assert verify_triad(counts, tampered) is False

    def test_verify_invalid_triad_wrong_c3(self) -> None:
        """Tampered c3 fails verification."""
        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad = compute_triad(counts)
        tampered = EditionTriad(
            c1=triad.c1,
            c2=triad.c2,
            c3=(triad.c3 + 1) % 97,
        )
        assert verify_triad(counts, tampered) is False


class TestTriadParsing:
    """Test triad parsing from compact format."""

    def test_parse_valid(self) -> None:
        """Parse valid compact format."""
        triad = parse_triad("37-45-17")
        assert triad.c1 == 37
        assert triad.c2 == 45
        assert triad.c3 == 17

    def test_parse_roundtrip(self) -> None:
        """Parse(triad.compact) == triad."""
        original = EditionTriad(c1=12, c2=34, c3=56)
        parsed = parse_triad(original.compact)
        assert parsed == original

    def test_parse_invalid_format(self) -> None:
        """Invalid format raises error."""
        with pytest.raises(ValueError):
            parse_triad("12-34")  # Missing c3

    def test_parse_invalid_values(self) -> None:
        """Out-of-range values raise error."""
        with pytest.raises(ValueError):
            parse_triad("12-34-100")  # c3 >= 97


class TestBase32Encoding:
    """Test Crockford Base32 encoding."""

    def test_encode_zero(self) -> None:
        """Encoding zero produces all zeros."""
        result = encode_base32(0, length=4)
        assert result == "0000"

    def test_encode_max_single_char(self) -> None:
        """Values 0-31 encode to single meaningful characters."""
        assert encode_base32(31, length=1) == "Z"

    def test_encode_round_trip(self) -> None:
        """Encode then decode returns original value."""
        for value in [0, 1, 31, 32, 100, 1000, 123456, 2**20 - 1]:
            encoded = encode_base32(value, length=8)
            decoded = decode_base32(encoded)
            assert decoded == value, f"Round-trip failed for {value}"

    def test_encode_deterministic(self) -> None:
        """Same value always encodes the same way."""
        v1 = encode_base32(12345, length=6)
        v2 = encode_base32(12345, length=6)
        assert v1 == v2

    def test_decode_case_insensitive(self) -> None:
        """Decoding is case-insensitive."""
        assert decode_base32("ABC") == decode_base32("abc")

    def test_decode_confusable_characters(self) -> None:
        """Crockford handles I/L/O ambiguity."""
        # I and L both map to 1, O maps to 0
        assert decode_base32("I") == decode_base32("1")
        assert decode_base32("L") == decode_base32("1")
        assert decode_base32("O") == decode_base32("0")


class TestEID12:
    """Test EID12 generation."""

    def test_eid12_format(self) -> None:
        """EID12 has correct format."""
        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad = compute_triad(counts)
        eid = triad_to_eid12(triad)
        assert isinstance(eid, str)
        # Format: PP-XXXX-XXXX (12 chars total)
        assert len(eid) == 12
        parts = eid.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 2  # prefix
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4

    def test_eid12_deterministic(self) -> None:
        """Same triad = same EID."""
        triad = EditionTriad(c1=37, c2=45, c3=17)
        eid1 = triad_to_eid12(triad)
        eid2 = triad_to_eid12(triad)
        assert eid1 == eid2

    def test_eid12_different_inputs(self) -> None:
        """Different triads produce different EIDs."""
        triad1 = EditionTriad(c1=37, c2=45, c3=17)
        triad2 = EditionTriad(c1=38, c2=45, c3=17)
        eid1 = triad_to_eid12(triad1)
        eid2 = triad_to_eid12(triad2)
        assert eid1 != eid2

    def test_eid12_custom_prefix(self) -> None:
        """Custom prefix is used."""
        triad = EditionTriad(c1=1, c2=2, c3=3)
        eid = triad_to_eid12(triad, case_prefix="AB")
        assert eid.startswith("AB-")


class TestPrimeProperties:
    """Test mathematical properties related to primes."""

    def test_modulus_is_97(self) -> None:
        """Modulus is the largest 2-digit prime."""
        assert TRIAD_MODULUS == 97

    def test_coefficients_are_prime(self) -> None:
        """C2 uses prime coefficients (except 1)."""
        # PRIME_COEFFICIENTS = (1, 2, 3, 5, 7)
        # 1 is not prime, but others are

        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

        # Coefficients 2,3,5,7 should be prime
        for coef in PRIME_COEFFICIENTS[1:]:
            assert is_prime(coef), f"{coef} is not prime"

    def test_coefficients_coprime_to_modulus(self) -> None:
        """All coefficients are coprime to 97."""
        import math

        for coef in PRIME_COEFFICIENTS:
            assert math.gcd(coef, TRIAD_MODULUS) == 1, f"{coef} not coprime to 97"

    def test_linear_independence_property(self) -> None:
        """Different inputs cannot produce same C2 by coincidence.

        With prime coefficients coprime to 97, collisions require
        large coordinated changes.
        """
        # If we change only one component, C2 must change
        base = EditionCounts(pages=10, figures=10, tables=10, equations=10, references=10)
        changed_p = EditionCounts(pages=11, figures=10, tables=10, equations=10, references=10)
        changed_f = EditionCounts(pages=10, figures=11, tables=10, equations=10, references=10)
        changed_t = EditionCounts(pages=10, figures=10, tables=11, equations=10, references=10)
        changed_e = EditionCounts(pages=10, figures=10, tables=10, equations=11, references=10)
        changed_r = EditionCounts(pages=10, figures=10, tables=10, equations=10, references=11)

        triads = [compute_triad(c) for c in [base, changed_p, changed_f, changed_t, changed_e, changed_r]]
        c2_values = {t.c2 for t in triads}

        # Not all the same (extremely unlikely with prime coefficients)
        assert len(c2_values) > 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_inputs(self) -> None:
        """All-zero inputs produce valid triad."""
        counts = EditionCounts(pages=0, figures=0, tables=0, equations=0, references=0)
        triad = compute_triad(counts)
        assert triad.c1 == 0
        assert triad.c2 == 0
        assert triad.c3 == 0

    def test_large_inputs(self) -> None:
        """Large inputs still produce valid triads."""
        counts = EditionCounts(
            pages=10**6,
            figures=10**5,
            tables=10**4,
            equations=10**3,
            references=10**6,
        )
        triad = compute_triad(counts)
        assert 0 <= triad.c1 < 97
        assert 0 <= triad.c2 < 97
        assert 0 <= triad.c3 < 97

    def test_triad_equality(self) -> None:
        """EditionTriad equality works correctly."""
        t1 = EditionTriad(1, 2, 3)
        t2 = EditionTriad(1, 2, 3)
        t3 = EditionTriad(1, 2, 4)
        assert t1 == t2
        assert t1 != t3

    def test_triad_hashable(self) -> None:
        """EditionTriad can be used in sets/dicts."""
        t1 = EditionTriad(1, 2, 3)
        t2 = EditionTriad(1, 2, 3)
        s = {t1, t2}
        assert len(s) == 1  # Same triad, only one entry

    def test_triad_compact_format(self) -> None:
        """Compact format is zero-padded."""
        triad = EditionTriad(c1=1, c2=2, c3=3)
        assert triad.compact == "01-02-03"

    def test_triad_invalid_range(self) -> None:
        """Out-of-range values raise error."""
        with pytest.raises(ValueError):
            EditionTriad(c1=100, c2=0, c3=0)
