"""Particle Catalog Closure — SM.INTSTACK.v1

Complete Standard Model particle table with mass, charge, spin, color
charge, and dimensionless UMCP coordinate embedding.

Physics:
  The Standard Model contains:
    6 quarks:   u, d, c, s, t, b       (spin-½, color-charged)
    6 leptons:  e, μ, τ, ν_e, ν_μ, ν_τ (spin-½, color-neutral)
    4 gauge bosons: γ, W±, Z⁰, g       (spin-1)
    1 scalar boson: H                    (spin-0)

UMCP integration:
  c1 = mass_embedding = log₁₀(m/m_ref) normalized to [0,1]
  c2 = charge_embedding = |Q/e| normalized
  c3 = spin_embedding = 2s / 2 normalized
  ω = 1 − stability_fraction (stable particles have ω → 0)
  F = 1 − ω

Cross-references:
  Contract:  contracts/SM.INTSTACK.v1.yaml
  Sources:   PDG 2024 (Particle Data Group)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ParticleCategory(StrEnum):
    QUARK = "Quark"
    LEPTON = "Lepton"
    GAUGE_BOSON = "GaugeBoson"
    SCALAR_BOSON = "ScalarBoson"


class StabilityRegime(StrEnum):
    STABLE = "Stable"
    LONG_LIVED = "LongLived"
    RESONANCE = "Resonance"
    VIRTUAL = "Virtual"


@dataclass(frozen=True)
class Particle:
    """Standard Model particle data."""

    name: str
    symbol: str
    category: str
    mass_GeV: float  # Rest mass in GeV/c²
    charge_e: float  # Electric charge in units of e
    spin: float  # Spin quantum number
    color_charge: bool  # Carries color charge
    generation: int  # 1, 2, or 3 (0 for bosons)
    antiparticle: str  # Name of antiparticle
    lifetime_s: float  # Mean lifetime (s), 0 = stable
    width_GeV: float  # Decay width (GeV), 0 = stable
    # UMCP coordinates
    c1: float  # Mass embedding [0,1]
    c2: float  # Charge embedding [0,1]
    c3: float  # Spin embedding [0,1]
    omega: float  # Stability drift
    F: float  # Stability fidelity
    regime: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ── Mass reference for log embedding ────────────────────────────
M_REF = 1e-10  # GeV (neutrino-scale floor)
M_MAX_LOG = math.log10(200.0 / M_REF)  # Higgs ~ 125 GeV, top ~ 173 GeV

# Lifetime reference
TAU_UNIVERSE_S = 4.35e17  # Age of universe (s)


def _mass_embed(m_gev: float) -> float:
    """Log-scale mass embedding to [0,1]."""
    if m_gev <= 0:
        return 0.0
    return min(1.0, max(0.0, math.log10(m_gev / M_REF) / M_MAX_LOG))


def _stability_omega(lifetime_s: float) -> float:
    """Map lifetime to stability drift ω ∈ [0,1]."""
    if lifetime_s <= 0:
        return 0.0  # stable
    if lifetime_s >= TAU_UNIVERSE_S:
        return 0.001  # effectively stable
    # Log-scale: shorter lifetime → higher ω
    ratio = math.log10(TAU_UNIVERSE_S / lifetime_s) / 30.0
    return min(1.0, max(0.0, ratio))


def _stability_regime(lifetime_s: float, width_gev: float) -> str:
    if lifetime_s <= 0 or lifetime_s >= TAU_UNIVERSE_S:
        return StabilityRegime.STABLE.value
    if lifetime_s > 1e-10:
        return StabilityRegime.LONG_LIVED.value
    if width_gev > 0.1:
        return StabilityRegime.VIRTUAL.value
    return StabilityRegime.RESONANCE.value


# ── Complete SM Particle Table (PDG 2024) ────────────────────────
_PARTICLES_RAW: list[dict[str, Any]] = [
    # Quarks (generation 1)
    {
        "name": "up",
        "symbol": "u",
        "category": "Quark",
        "mass_GeV": 0.00216,
        "charge_e": 2 / 3,
        "spin": 0.5,
        "color_charge": True,
        "generation": 1,
        "antiparticle": "anti-up",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    {
        "name": "down",
        "symbol": "d",
        "category": "Quark",
        "mass_GeV": 0.00467,
        "charge_e": -1 / 3,
        "spin": 0.5,
        "color_charge": True,
        "generation": 1,
        "antiparticle": "anti-down",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    # Quarks (generation 2)
    {
        "name": "charm",
        "symbol": "c",
        "category": "Quark",
        "mass_GeV": 1.27,
        "charge_e": 2 / 3,
        "spin": 0.5,
        "color_charge": True,
        "generation": 2,
        "antiparticle": "anti-charm",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    {
        "name": "strange",
        "symbol": "s",
        "category": "Quark",
        "mass_GeV": 0.093,
        "charge_e": -1 / 3,
        "spin": 0.5,
        "color_charge": True,
        "generation": 2,
        "antiparticle": "anti-strange",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    # Quarks (generation 3)
    {
        "name": "top",
        "symbol": "t",
        "category": "Quark",
        "mass_GeV": 172.69,
        "charge_e": 2 / 3,
        "spin": 0.5,
        "color_charge": True,
        "generation": 3,
        "antiparticle": "anti-top",
        "lifetime_s": 5e-25,
        "width_GeV": 1.42,
    },
    {
        "name": "bottom",
        "symbol": "b",
        "category": "Quark",
        "mass_GeV": 4.18,
        "charge_e": -1 / 3,
        "spin": 0.5,
        "color_charge": True,
        "generation": 3,
        "antiparticle": "anti-bottom",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    # Leptons (generation 1)
    {
        "name": "electron",
        "symbol": "e⁻",
        "category": "Lepton",
        "mass_GeV": 0.000511,
        "charge_e": -1.0,
        "spin": 0.5,
        "color_charge": False,
        "generation": 1,
        "antiparticle": "positron",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    {
        "name": "electron neutrino",
        "symbol": "ν_e",
        "category": "Lepton",
        "mass_GeV": 1e-10,
        "charge_e": 0.0,
        "spin": 0.5,
        "color_charge": False,
        "generation": 1,
        "antiparticle": "anti-electron neutrino",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    # Leptons (generation 2)
    {
        "name": "muon",
        "symbol": "μ⁻",
        "category": "Lepton",
        "mass_GeV": 0.10566,
        "charge_e": -1.0,
        "spin": 0.5,
        "color_charge": False,
        "generation": 2,
        "antiparticle": "anti-muon",
        "lifetime_s": 2.197e-6,
        "width_GeV": 3e-19,
    },
    {
        "name": "muon neutrino",
        "symbol": "ν_μ",
        "category": "Lepton",
        "mass_GeV": 1e-10,
        "charge_e": 0.0,
        "spin": 0.5,
        "color_charge": False,
        "generation": 2,
        "antiparticle": "anti-muon neutrino",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    # Leptons (generation 3)
    {
        "name": "tau",
        "symbol": "τ⁻",
        "category": "Lepton",
        "mass_GeV": 1.777,
        "charge_e": -1.0,
        "spin": 0.5,
        "color_charge": False,
        "generation": 3,
        "antiparticle": "anti-tau",
        "lifetime_s": 2.903e-13,
        "width_GeV": 2.27e-12,
    },
    {
        "name": "tau neutrino",
        "symbol": "ν_τ",
        "category": "Lepton",
        "mass_GeV": 1e-10,
        "charge_e": 0.0,
        "spin": 0.5,
        "color_charge": False,
        "generation": 3,
        "antiparticle": "anti-tau neutrino",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    # Gauge Bosons
    {
        "name": "photon",
        "symbol": "γ",
        "category": "GaugeBoson",
        "mass_GeV": 0.0,
        "charge_e": 0.0,
        "spin": 1.0,
        "color_charge": False,
        "generation": 0,
        "antiparticle": "photon",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    {
        "name": "W boson",
        "symbol": "W±",
        "category": "GaugeBoson",
        "mass_GeV": 80.377,
        "charge_e": 1.0,
        "spin": 1.0,
        "color_charge": False,
        "generation": 0,
        "antiparticle": "W∓",
        "lifetime_s": 3.17e-25,
        "width_GeV": 2.085,
    },
    {
        "name": "Z boson",
        "symbol": "Z⁰",
        "category": "GaugeBoson",
        "mass_GeV": 91.1876,
        "charge_e": 0.0,
        "spin": 1.0,
        "color_charge": False,
        "generation": 0,
        "antiparticle": "Z⁰",
        "lifetime_s": 2.64e-25,
        "width_GeV": 2.4952,
    },
    {
        "name": "gluon",
        "symbol": "g",
        "category": "GaugeBoson",
        "mass_GeV": 0.0,
        "charge_e": 0.0,
        "spin": 1.0,
        "color_charge": True,
        "generation": 0,
        "antiparticle": "gluon",
        "lifetime_s": 0,
        "width_GeV": 0,
    },
    # Scalar Boson
    {
        "name": "Higgs boson",
        "symbol": "H⁰",
        "category": "ScalarBoson",
        "mass_GeV": 125.25,
        "charge_e": 0.0,
        "spin": 0.0,
        "color_charge": False,
        "generation": 0,
        "antiparticle": "Higgs boson",
        "lifetime_s": 1.56e-22,
        "width_GeV": 4.07e-3,
    },
]


def _build_particle(raw: dict[str, Any]) -> Particle:
    m = raw["mass_GeV"]
    lt = raw["lifetime_s"]
    w = raw["width_GeV"]
    omega = _stability_omega(lt)
    return Particle(
        name=raw["name"],
        symbol=raw["symbol"],
        category=raw["category"],
        mass_GeV=m,
        charge_e=raw["charge_e"],
        spin=raw["spin"],
        color_charge=raw["color_charge"],
        generation=raw["generation"],
        antiparticle=raw["antiparticle"],
        lifetime_s=lt,
        width_GeV=w,
        c1=round(_mass_embed(m), 6),
        c2=round(min(1.0, abs(raw["charge_e"])), 6),
        c3=round(min(1.0, raw["spin"]), 6),
        omega=round(omega, 6),
        F=round(1.0 - omega, 6),
        regime=_stability_regime(lt, w),
    )


# Build catalog
SM_CATALOG: dict[str, Particle] = {}
for _raw in _PARTICLES_RAW:
    p = _build_particle(_raw)
    SM_CATALOG[p.name] = p


def get_particle(name: str) -> Particle:
    """Get a Standard Model particle by name."""
    key = name.lower()
    for pname, p in SM_CATALOG.items():
        if pname.lower() == key or p.symbol.lower() == key:
            return p
    msg = f"Unknown particle: {name}. Available: {list(SM_CATALOG.keys())}"
    raise KeyError(msg)


def list_particles(category: str | None = None) -> list[Particle]:
    """List all SM particles, optionally filtered by category."""
    if category is None:
        return list(SM_CATALOG.values())
    return [p for p in SM_CATALOG.values() if p.category.lower() == category.lower()]


def particle_table() -> list[dict[str, Any]]:
    """Return full particle table as list of dicts for dashboard display."""
    return [p.to_dict() for p in SM_CATALOG.values()]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    for cat in ["Quark", "Lepton", "GaugeBoson", "ScalarBoson"]:
        print(f"\n{'─' * 60}")
        print(f"  {cat}s")
        print(f"{'─' * 60}")
        for p in list_particles(cat):
            print(
                f"  {p.symbol:6s}  m={p.mass_GeV:12.6f} GeV  Q={p.charge_e:+5.2f}e  "
                f"s={p.spin:.1f}  ω={p.omega:.4f}  {p.regime}"
            )
