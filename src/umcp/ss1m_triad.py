"""
SS1m Edition Triad Checksums

Implements the mod-97 prime-field checksums for human-verifiable edition identity.
Based on "The Episteme of Return" and SS1m specification.

The triad (C1, C2, C3) provides a compact, human-checkable fingerprint for
publication editions using prime-field arithmetic.

Mathematical Foundation:
    - Modulus 97 (largest 2-digit prime) for human readability
    - Prime coefficients {1, 2, 3, 5, 7} ensure linear independence
    - Three checksums provide redundancy for error detection

Canonical Formulas (The Episteme of Return):
    C1 = (P + F + T + E + R) mod 97          # Sum checksum
    C2 = (P + 2F + 3T + 5E + 7R) mod 97      # Weighted checksum
    C3 = (P·F + T·E + R) mod 97              # Product checksum

Where:
    P = page count
    F = figure count
    T = table count
    E = equation count
    R = reference count

Reference: SeamStamp v1-mini (SS1m), Clement Paulus, October 25, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# The verification prime - largest 2-digit prime
TRIAD_MODULUS = 97

# Prime coefficients for C2 weighted checksum
PRIME_COEFFICIENTS = (1, 2, 3, 5, 7)  # For P, F, T, E, R


class EditionCounts(NamedTuple):
    """Edition component counts for triad computation."""

    pages: int  # P
    figures: int  # F
    tables: int  # T
    equations: int  # E
    references: int  # R


@dataclass(frozen=True)
class EditionTriad:
    """
    The (C1, C2, C3) checksum triad for edition identity.

    Properties:
        c1: Simple additive checksum (P + F) mod 97
        c2: Weighted checksum with prime coefficients
        c3: Multiplicative checksum (P * F + T) mod 97
        compact: Two-digit string representation "C1-C2-C3"
    """

    c1: int
    c2: int
    c3: int

    def __post_init__(self) -> None:
        """Validate all components are in valid range."""
        for name, val in [("c1", self.c1), ("c2", self.c2), ("c3", self.c3)]:
            if not 0 <= val < TRIAD_MODULUS:
                raise ValueError(f"{name}={val} not in [0, {TRIAD_MODULUS})")

    @property
    def compact(self) -> str:
        """Compact string representation: 'CC-CC-CC' format."""
        return f"{self.c1:02d}-{self.c2:02d}-{self.c3:02d}"

    def __str__(self) -> str:
        return self.compact


def compute_triad(counts: EditionCounts) -> EditionTriad:
    """
    Compute the edition identity triad from component counts.

    Args:
        counts: EditionCounts with (pages, figures, tables, equations, references)

    Returns:
        EditionTriad with (c1, c2, c3) checksums

    Example:
        >>> counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        >>> triad = compute_triad(counts)
        >>> print(triad.compact)
        '37-45-17'
    """
    P, F, T, E, R = counts

    # C1: Sum checksum (P + F + T + E + R) mod 97
    # Canonical: The Episteme of Return
    c1 = (P + F + T + E + R) % TRIAD_MODULUS

    # C2: Weighted sum with prime coefficients
    # C2 = (1*P + 2*F + 3*T + 5*E + 7*R) mod 97
    c2 = (
        PRIME_COEFFICIENTS[0] * P
        + PRIME_COEFFICIENTS[1] * F
        + PRIME_COEFFICIENTS[2] * T
        + PRIME_COEFFICIENTS[3] * E
        + PRIME_COEFFICIENTS[4] * R
    ) % TRIAD_MODULUS

    # C3: Product checksum (P·F + T·E + R) mod 97
    # Canonical: The Episteme of Return
    c3 = (P * F + T * E + R) % TRIAD_MODULUS

    return EditionTriad(c1=c1, c2=c2, c3=c3)


def verify_triad(counts: EditionCounts, expected: EditionTriad) -> bool:
    """
    Verify that counts produce the expected triad.

    Args:
        counts: Edition component counts
        expected: Expected triad values

    Returns:
        True if computed triad matches expected
    """
    computed = compute_triad(counts)
    return computed == expected


def parse_triad(compact: str) -> EditionTriad:
    """
    Parse a compact triad string 'CC-CC-CC' into EditionTriad.

    Args:
        compact: String in format 'DD-DD-DD' where DD is 00-96

    Returns:
        EditionTriad

    Raises:
        ValueError: If format is invalid or values out of range
    """
    parts = compact.split("-")
    if len(parts) != 3:
        raise ValueError(f"Expected 'CC-CC-CC' format, got '{compact}'")

    try:
        c1, c2, c3 = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as e:
        raise ValueError(f"Non-integer component in '{compact}'") from e

    return EditionTriad(c1=c1, c2=c2, c3=c3)


# Crockford Base32 encoding for EID12 (mod 32, not prime)
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def encode_base32(value: int, length: int = 1) -> str:
    """
    Encode an integer to Crockford Base32.

    Args:
        value: Non-negative integer to encode
        length: Minimum output length (zero-padded)

    Returns:
        Crockford Base32 string
    """
    if value < 0:
        raise ValueError("Cannot encode negative values")

    if value == 0:
        return CROCKFORD_ALPHABET[0] * length

    result = []
    while value > 0:
        result.append(CROCKFORD_ALPHABET[value % 32])
        value //= 32

    result.reverse()
    encoded = "".join(result)

    # Pad to minimum length
    if len(encoded) < length:
        encoded = CROCKFORD_ALPHABET[0] * (length - len(encoded)) + encoded

    return encoded


def decode_base32(encoded: str) -> int:
    """
    Decode a Crockford Base32 string to integer.

    Args:
        encoded: Crockford Base32 string

    Returns:
        Decoded integer value
    """
    # Normalize: uppercase, handle common substitutions
    normalized = encoded.upper()
    normalized = normalized.replace("O", "0").replace("I", "1").replace("L", "1")

    value = 0
    for char in normalized:
        idx = CROCKFORD_ALPHABET.find(char)
        if idx < 0:
            raise ValueError(f"Invalid Base32 character: '{char}'")
        value = value * 32 + idx

    return value


def triad_to_eid12(triad: EditionTriad, case_prefix: str = "CP") -> str:
    """
    Convert triad to compact EID12 format.

    Format: PREFIX-XXXX-XXXX (12 characters total)
    Uses Crockford Base32 for compactness.

    Args:
        triad: EditionTriad to encode
        case_prefix: 2-character case prefix

    Returns:
        EID12 string
    """
    if len(case_prefix) != 2:
        raise ValueError("Case prefix must be exactly 2 characters")

    # Pack triad into single value: c1 * 97^2 + c2 * 97 + c3
    packed = triad.c1 * (TRIAD_MODULUS**2) + triad.c2 * TRIAD_MODULUS + triad.c3

    # Encode as Base32 (4 chars each for two groups)
    encoded = encode_base32(packed, length=8)

    return f"{case_prefix}-{encoded[:4]}-{encoded[4:]}"


if __name__ == "__main__":
    # Example from manuscript: P=25, Eq=12, Fig=5, Tab=3, List=2, Box=1, Ref=48
    # Mapping to our schema: P=25, F=5, T=3, E=12, R=48
    example = EditionCounts(pages=25, figures=5, tables=3, equations=12, references=48)
    triad = compute_triad(example)

    print("SS1m Edition Triad Checksum")
    print("=" * 40)
    print(
        f"Input: P={example.pages}, F={example.figures}, T={example.tables}, E={example.equations}, R={example.references}"
    )
    print(f"Modulus: {TRIAD_MODULUS} (prime)")
    print(f"Coefficients: {PRIME_COEFFICIENTS} (primes)")
    print()
    print(f"C1 = ({example.pages} + {example.figures}) mod 97 = {triad.c1}")
    print(
        f"C2 = (1×{example.pages} + 2×{example.figures} + 3×{example.tables} + 5×{example.equations} + 7×{example.references}) mod 97 = {triad.c2}"
    )
    print(f"C3 = ({example.pages} × {example.figures} + {example.tables}) mod 97 = {triad.c3}")
    print()
    print(f"Triad: {triad.compact}")
    print(f"EID12: {triad_to_eid12(triad, 'CP')}")
