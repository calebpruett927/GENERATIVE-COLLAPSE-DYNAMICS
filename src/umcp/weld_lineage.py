"""
Weld Lineage — Formal Separation of Continuity Claims from Structural Fingerprints

Implements the paper's explicit distinction between:
    - Edition Identity (EID): Structural fingerprint of a document
      (counts of pages, equations, figures, tables, listings, boxes, references)
    - Weld Lineage: Continuity claim between two editions under a frozen contract

This separation is intentional and prevents the common failure mode where
revisions silently change definitions while still being presented as "the same"
result.

A Weld records:
    - The PRE anchor (prior edition EID + canon reference)
    - The POST edition (current EID)
    - The budget identity closure: Δκ, ir, residual
    - The seam verdict: PASS / FAIL

An EID records:
    - Component counts (P, Eq, Fig, Tab, List, Box, Ref)
    - Triad checksums (C1, C2, C3) per SS1m specification
    - Compact EID12 encoding

The weld-lineage separation guarantees that:
    1. A revision's identity and its continuity status are independent things
    2. Two editions with identical EIDs are structurally identical (same counts)
    3. A passing weld means the transition preserves κ-continuity
    4. History is append-only and welded, never rewritten
       (*Historia numquam rescribitur; sutura tantum additur.*)

Reference: The Physics of Coherence, §Edition Identity vs Weld Lineage
Cross-references:
    - ss1m_triad.py        (EditionTriad, EditionCounts — 5-field legacy)
    - continuity_law.py    (ContinuityVerdict, verify_continuity_law)
    - frozen_contract.py   (check_seam_pass)
    - PUBLICATION_INFRASTRUCTURE docs
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NamedTuple

from .ss1m_triad import TRIAD_MODULUS, EditionTriad

# =============================================================================
# 7-FIELD EDITION IDENTITY (per "The Physics of Coherence")
# =============================================================================


class EditionIdentity(NamedTuple):
    """7-field structural fingerprint per the paper's specification.

    EID = (P, Eq, Fig, Tab, List, Box, Ref)

    The paper's own EID:
        P=34, Eq=52, Fig=3, Tab=6, List=4, Box=9, Ref=12

    These counts are author-frozen per revision. Changing any count
    produces a different EID, regardless of whether the weld passes.
    """

    pages: int  # P
    equations: int  # Eq
    figures: int  # Fig
    tables: int  # Tab
    listings: int  # List
    boxes: int  # Box
    references: int  # Ref


# Extended prime coefficients for 7-field triad: (1, 2, 3, 5, 7, 11, 13)
EXTENDED_PRIME_COEFFICIENTS = (1, 2, 3, 5, 7, 11, 13)


def compute_extended_triad(eid: EditionIdentity) -> EditionTriad:
    """
    Compute EID triad from 7-field edition identity.

    Extended formulas (mod 97):
        C1 = (P + Eq + Fig + Tab + List + Box + Ref) mod 97
        C2 = (1P + 2Eq + 3Fig + 5Tab + 7List + 11Box + 13Ref) mod 97
        C3 = (P·Eq + Fig·Tab + List·Box + Ref) mod 97

    Args:
        eid: 7-field EditionIdentity

    Returns:
        EditionTriad with checksums
    """
    P, Eq, Fig, Tab, List, Box, Ref = eid

    # C1: Sum checksum
    c1 = (P + Eq + Fig + Tab + List + Box + Ref) % TRIAD_MODULUS

    # C2: Weighted sum with extended prime coefficients
    c2 = sum(coeff * val for coeff, val in zip(EXTENDED_PRIME_COEFFICIENTS, eid, strict=True)) % TRIAD_MODULUS

    # C3: Product checksum (pairwise products + terminal)
    c3 = (P * Eq + Fig * Tab + List * Box + Ref) % TRIAD_MODULUS

    return EditionTriad(c1=c1, c2=c2, c3=c3)


def verify_extended_triad(eid: EditionIdentity, expected: EditionTriad) -> bool:
    """Verify that 7-field EID produces the expected triad."""
    return compute_extended_triad(eid) == expected


# =============================================================================
# SS1m RECEIPT (full receipt with Δκ, ir, M0, M1, chk)
# =============================================================================


@dataclass(frozen=True)
class SS1mReceipt:
    """Full SS1m receipt as specified in the paper.

    The paper's receipt format:
        SS1m(EID) | Δκ = 0.5355182364 | ir = 1.7083333333
        | M0 = 205, M1 = 120 | EID 1
        : P=34, Eq=52, Fig=3, Tab=6, List=4, Box=9, Ref=12
        | chk=[68, 91, 12]

    Fields:
        eid: 7-field edition identity
        delta_kappa: Budget identity value
        ir: Interpretive density ratio (I₁/I₀)
        M0: Mass count 0 (total component mass before)
        M1: Mass count 1 (total component mass after)
        triad: Checksum triad [C1, C2, C3]
    """

    eid: EditionIdentity
    delta_kappa: float
    ir: float
    M0: int
    M1: int
    triad: EditionTriad

    @property
    def chk(self) -> list[int]:
        """Checksum list [C1, C2, C3]."""
        return [self.triad.c1, self.triad.c2, self.triad.c3]

    def compact(self) -> str:
        """Compact SS1m format string."""
        return (
            f"SS1m(EID) | Δκ = {self.delta_kappa:.10f} "
            f"| ir = {self.ir:.10f} "
            f"| M0 = {self.M0}, M1 = {self.M1} "
            f"| EID 1: P={self.eid.pages}, Eq={self.eid.equations}, "
            f"Fig={self.eid.figures}, Tab={self.eid.tables}, "
            f"List={self.eid.listings}, Box={self.eid.boxes}, "
            f"Ref={self.eid.references} "
            f"| chk={self.chk}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "eid": {
                "pages": self.eid.pages,
                "equations": self.eid.equations,
                "figures": self.eid.figures,
                "tables": self.eid.tables,
                "listings": self.eid.listings,
                "boxes": self.eid.boxes,
                "references": self.eid.references,
            },
            "delta_kappa": self.delta_kappa,
            "ir": self.ir,
            "M0": self.M0,
            "M1": self.M1,
            "triad": {
                "c1": self.triad.c1,
                "c2": self.triad.c2,
                "c3": self.triad.c3,
                "compact": self.triad.compact,
            },
        }

    def __str__(self) -> str:
        return self.compact()


def compute_ss1m_receipt(
    eid: EditionIdentity,
    delta_kappa: float,
    ir: float,
    M0: int,
    M1: int,
) -> SS1mReceipt:
    """
    Compute a full SS1m receipt from edition identity and budget values.

    Args:
        eid: 7-field edition identity
        delta_kappa: Budget identity value Δκ
        ir: Interpretive density ratio I₁/I₀
        M0: Mass count 0 (total component mass before)
        M1: Mass count 1 (total component mass after)

    Returns:
        SS1mReceipt with computed triad
    """
    triad = compute_extended_triad(eid)
    return SS1mReceipt(
        eid=eid,
        delta_kappa=delta_kappa,
        ir=ir,
        M0=M0,
        M1=M1,
        triad=triad,
    )


# =============================================================================
# WELD LINEAGE (continuity claim structure)
# =============================================================================


@dataclass(frozen=True)
class WeldAnchor:
    """Anchor point for a weld (PRE or POST edition).

    Represents one side of the weld — either the prior artifact
    (PRE anchor) or the current artifact (POST edition).
    """

    eid: EditionIdentity
    triad: EditionTriad
    canon_ref: str  # e.g., "paulus2025episteme"
    artifact_sha256: str  # SHA-256 of the source PDF or artifact
    timestamp_utc: str  # ISO 8601 timestamp


@dataclass(frozen=True)
class WeldLineage:
    """Continuity claim between two editions under a frozen contract.

    A weld is not a stylistic label for a revision. It is a continuity
    claim with explicit admissibility conditions: the PRE→POST transition
    must satisfy the declared budget identity and seam residual constraints
    under a frozen contract.

    The weld is logically independent of the EID:
    - Two editions with identical EIDs may fail the weld (broken continuity)
    - Two editions with different EIDs may pass the weld (valid evolution)

    *Historia numquam rescribitur; sutura tantum additur.*
    """

    weld_id: str  # e.g., "W-2025-12-31-PHYS-COHERENCE"
    pre_anchor: WeldAnchor
    post_anchor: WeldAnchor
    delta_kappa: float  # Budget identity value
    ir: float  # Interpretive density ratio
    residual: float  # Seam residual |s|
    seam_pass: bool  # Whether the weld closes
    failures: tuple[str, ...] = ()
    created_utc: str = ""

    def __post_init__(self) -> None:
        if not self.created_utc:
            object.__setattr__(self, "created_utc", datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON/YAML output."""
        return {
            "weld_id": self.weld_id,
            "pre_anchor": {
                "eid": dict(self.pre_anchor.eid._asdict()),
                "triad": self.pre_anchor.triad.compact,
                "canon_ref": self.pre_anchor.canon_ref,
                "artifact_sha256": self.pre_anchor.artifact_sha256,
                "timestamp_utc": self.pre_anchor.timestamp_utc,
            },
            "post_anchor": {
                "eid": dict(self.post_anchor.eid._asdict()),
                "triad": self.post_anchor.triad.compact,
                "canon_ref": self.post_anchor.canon_ref,
                "artifact_sha256": self.post_anchor.artifact_sha256,
                "timestamp_utc": self.post_anchor.timestamp_utc,
            },
            "delta_kappa": self.delta_kappa,
            "ir": self.ir,
            "residual": self.residual,
            "seam_pass": self.seam_pass,
            "failures": list(self.failures),
            "created_utc": self.created_utc,
        }


def create_weld(
    weld_id: str,
    pre_anchor: WeldAnchor,
    post_anchor: WeldAnchor,
    delta_kappa: float,
    ir: float,
    residual: float,
    tol_seam: float = 0.005,
    tol_identity: float = 1e-6,
) -> WeldLineage:
    """
    Create a weld lineage record with automatic pass/fail determination.

    Args:
        weld_id: Weld identifier
        pre_anchor: PRE anchor (prior edition)
        post_anchor: POST anchor (current edition)
        delta_kappa: Budget identity value
        ir: Interpretive density ratio
        residual: Seam residual
        tol_seam: Seam tolerance (frozen)
        tol_identity: Identity check tolerance

    Returns:
        WeldLineage with computed pass/fail status
    """
    import math

    failures: list[str] = []

    # Check residual
    if abs(residual) > tol_seam:
        failures.append(f"|s| = {abs(residual):.6f} > tol_seam = {tol_seam}")

    # Check exponential identity
    ir_expected = math.exp(delta_kappa)
    identity_error = abs(ir - ir_expected)
    if identity_error > tol_identity:
        failures.append(f"|ir − exp(Δκ)| = {identity_error:.6e} > tol_id = {tol_identity}")

    return WeldLineage(
        weld_id=weld_id,
        pre_anchor=pre_anchor,
        post_anchor=post_anchor,
        delta_kappa=delta_kappa,
        ir=ir,
        residual=residual,
        seam_pass=len(failures) == 0,
        failures=tuple(failures),
    )
