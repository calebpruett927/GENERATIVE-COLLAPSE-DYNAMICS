/**
 * Physical Signature Library — Maps kernel invariants to real-world matter.
 *
 * Every known entity (particle, element, compound, material, state) has a
 * kernel signature: (F, ω, IC, Δ, S, C). This library:
 *   1. Stores ~80 reference entities across 7 physical scales
 *   2. Computes weighted distance between any kernel result and all references
 *   3. Classifies unknown traces by scale, structure, and composition
 *   4. Predicts physical properties from kernel similarity
 *
 * Tier-0 Protocol: no Tier-1 symbol is redefined. All frozen parameters
 * from constants.ts. Physical data from PDG 2024, NIST, and domain closures.
 */

import { computeKernel, classifyRegime, type KernelResult, type RegimeLabel } from './kernel';
import { EPSILON, REGIME_THRESHOLDS } from './constants';

/* ─── Scale Classification ──────────────────────────────────────── */

export type PhysicalScale =
  | 'subatomic'    // quarks, leptons, gauge bosons (< 1 fm)
  | 'hadronic'     // protons, neutrons, mesons (~ 1 fm)
  | 'nuclear'      // nuclei, binding, fission/fusion (1-10 fm)
  | 'atomic'       // atoms, electron shells (~ 100 pm)
  | 'molecular'    // molecules, compounds (~ 1 nm)
  | 'bulk'         // materials, crystals, fluids (> 1 μm)
  | 'astronomical' // stars, compact objects, galaxies (> 10⁶ m)
  | 'abstract';    // non-physical: market, consciousness, language

export interface ScaleInfo {
  scale: PhysicalScale;
  label: string;
  sizeRange: string;
  energyRange: string;
  description: string;
  color: string;        // Tailwind text color class
}

export const SCALE_INFO: Record<PhysicalScale, ScaleInfo> = {
  subatomic: {
    scale: 'subatomic', label: 'Subatomic', sizeRange: '< 10⁻¹⁸ m',
    energyRange: '0.5 MeV – 173 GeV', description: 'Fundamental particles: quarks, leptons, bosons',
    color: 'text-violet-400',
  },
  hadronic: {
    scale: 'hadronic', label: 'Hadronic', sizeRange: '~ 10⁻¹⁵ m (1 fm)',
    energyRange: '0.14 – 10 GeV', description: 'Composite particles bound by color confinement',
    color: 'text-red-400',
  },
  nuclear: {
    scale: 'nuclear', label: 'Nuclear', sizeRange: '1 – 10 fm',
    energyRange: '1 – 9 MeV/nucleon', description: 'Nuclei bound by the strong nuclear force',
    color: 'text-orange-400',
  },
  atomic: {
    scale: 'atomic', label: 'Atomic', sizeRange: '30 – 300 pm',
    energyRange: '4 – 25 eV (ionization)', description: 'Atoms with electron shells, periodic properties',
    color: 'text-yellow-400',
  },
  molecular: {
    scale: 'molecular', label: 'Molecular', sizeRange: '0.1 – 100 nm',
    energyRange: '0.1 – 10 eV (bond)', description: 'Molecules, compounds, crystal unit cells',
    color: 'text-green-400',
  },
  bulk: {
    scale: 'bulk', label: 'Bulk Material', sizeRange: '> 1 μm',
    energyRange: 'kJ/mol – MJ/mol', description: 'Macroscopic matter phases and engineered materials',
    color: 'text-cyan-400',
  },
  astronomical: {
    scale: 'astronomical', label: 'Astronomical', sizeRange: '> 10⁶ m',
    energyRange: '10²⁶ – 10⁴⁶ J', description: 'Stars, compact objects, large-scale structure',
    color: 'text-blue-400',
  },
  abstract: {
    scale: 'abstract', label: 'Abstract / Emergent', sizeRange: 'N/A',
    energyRange: 'N/A', description: 'Non-physical systems mapped via Tier-246 closure modules',
    color: 'text-kernel-400',
  },
};

/* ─── Physical Entity ───────────────────────────────────────────── */

export interface PhysicalEntity {
  name: string;
  symbol?: string;              // Chemical symbol or particle symbol
  scale: PhysicalScale;
  category: string;             // e.g., 'Noble gas', 'Baryon', 'Transition metal'
  c: number[];                  // 8-channel trace
  w: number[];                  // weights

  // Physical properties
  mass: string;                 // Human-readable mass string
  size: string;                 // Characteristic size
  composition: string;          // What it's made of
  structure: string;            // Structural description
  stability: string;            // Stability / lifetime

  // Kernel signature (computed lazily)
  _kernel?: KernelResult;

  // Interpretive notes
  signatureNote: string;        // What the kernel numbers mean for this entity
  keyChannels: string;          // Which channels dominate / kill IC
}

/* ─── The Reference Library ─────────────────────────────────────── */

export const PHYSICAL_ENTITIES: PhysicalEntity[] = [
  // ─── SUBATOMIC: Fundamental particles ───────────────────────────
  {
    name: 'Electron', symbol: 'e⁻', scale: 'subatomic', category: 'Lepton',
    c: [0.37, 1.0, 0.67, 0.0, 0.75, 1.0, 0.0, 0.33], w: Array(8).fill(0.125),
    mass: '0.511 MeV/c²', size: '< 10⁻¹⁸ m (point-like)', composition: 'Fundamental — no substructure',
    structure: 'Point particle, spin-½ fermion, charge −1e', stability: 'Stable (> 6.6×10²⁸ years)',
    signatureNote: 'Zero color and baryon channels (leptons carry no color charge). High charge and lepton number channels give moderate F but IC is crushed by the dead channels — geometric slaughter from color=0.',
    keyChannels: 'Color channel (c[3]=0) and baryon number (c[6]=0) kill IC.',
  },
  {
    name: 'Up Quark', symbol: 'u', scale: 'subatomic', category: 'Quark',
    c: [0.20, 0.67, 0.50, 1.0, 0.75, 0.33, 0.33, 0.33], w: Array(8).fill(0.125),
    mass: '2.16 MeV/c²', size: '< 10⁻¹⁸ m (point-like)', composition: 'Fundamental — no substructure',
    structure: 'Point particle, spin-½ fermion, charge +⅔e, color triplet', stability: 'Confined — never observed free',
    signatureNote: 'Low mass channel (lightest quark), but nonzero across all 8 channels. Color=1.0 distinguishes quarks from leptons. Generation=1/3 marks first generation.',
    keyChannels: 'Color channel (c[3]=1.0) is maximal — quarks ARE color charge carriers.',
  },
  {
    name: 'Top Quark', symbol: 't', scale: 'subatomic', category: 'Quark',
    c: [0.73, 0.50, 0.44, 1.0, 0.75, 0.0, 0.33, 1.0], w: Array(8).fill(0.125),
    mass: '172.69 GeV/c²', size: '< 10⁻¹⁸ m', composition: 'Fundamental — no substructure',
    structure: 'Spin-½ fermion, charge +⅔e, 3rd generation, decays before hadronization', stability: 'τ ≈ 5×10⁻²⁵ s — decays so fast it never forms hadrons',
    signatureNote: 'Highest mass channel of any quark. Generation channel at max (3/3). Stability channel near zero because it decays almost instantly. The only quark massive enough to decay before confinement acts.',
    keyChannels: 'Mass (c[0]=0.73) and generation (c[7]=1.0) are dominant. Stability (c[5]=0) kills IC.',
  },
  {
    name: 'Higgs Boson', symbol: 'H⁰', scale: 'subatomic', category: 'Scalar boson',
    c: [0.72, 0.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.0], w: Array(8).fill(0.125),
    mass: '125.25 GeV/c²', size: 'N/A (scalar field excitation)', composition: 'Fundamental — excitation of the Higgs field',
    structure: 'Spin-0 boson (unique scalar), no color, no charge, no generation', stability: 'τ ≈ 1.56×10⁻²² s, Γ ≈ 4.07 MeV',
    signatureNote: 'Maximally sparse: 6 of 8 channels at ε. Only mass and weak isospin nonzero. This gives extremely low IC (geometric mean of mostly zeros) despite moderate F. The Higgs signature is distinctive: high mass, nothing else.',
    keyChannels: 'Only c[0]=0.72 (mass) and c[4]=0.50 (weak) are nonzero. 6 dead channels → IC ≈ 0.',
  },
  {
    name: 'Photon', symbol: 'γ', scale: 'subatomic', category: 'Gauge boson',
    c: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], w: Array(8).fill(0.125),
    mass: '0', size: 'N/A (massless)', composition: 'Fundamental — electromagnetic force carrier',
    structure: 'Massless spin-1 boson, transverse polarization only', stability: 'Stable',
    signatureNote: 'Extremely sparse: mass=0, charge=0, color=0, but spin=1.0 and stability=1.0. The photon is the ultimate massless stable carrier — its signature has exactly 2 high channels and 6 dead ones.',
    keyChannels: 'Spin (c[2]=1.0) and stability (c[7]=1.0) are the only signal. 6 dead channels → IC ≈ 0.',
  },
  {
    name: 'W Boson', symbol: 'W±', scale: 'subatomic', category: 'Gauge boson',
    c: [0.71, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], w: Array(8).fill(0.125),
    mass: '80.377 GeV/c²', size: '< 10⁻¹⁸ m', composition: 'Fundamental — weak force carrier',
    structure: 'Massive spin-1 boson, carries weak charge and electric charge', stability: 'τ ≈ 3×10⁻²⁵ s',
    signatureNote: 'Heavy boson with charge and spin maximal, but color=0, stability=0, generation=0. The dead channels create a sharp heterogeneity gap. High F from mass+charge+spin, but IC collapses.',
    keyChannels: 'c[3]=0 (color), c[5]=0 (stability), c[6]=0 (generation) kill IC.',
  },

  // ─── HADRONIC: Composite particles ──────────────────────────────
  {
    name: 'Proton', symbol: 'p', scale: 'hadronic', category: 'Baryon',
    c: [0.60, 0.50, 1e-8, 1.0, 1e-8, 0.0, 0.33, 0.33], w: Array(8).fill(0.125),
    mass: '938.272 MeV/c² (0.9383 GeV)', size: '~ 0.87 fm (charge radius)', composition: 'uud — two up quarks and one down quark',
    structure: 'Spin-½ baryon, charge +1e, color singlet (confined). 99% of mass from QCD binding energy.',
    stability: 'Stable (τ > 10³⁴ years)',
    signatureNote: 'Confinement signature: color channel goes to ε (color is confined inside, not visible from outside). This single dead channel destroys IC via geometric slaughter — IC/F drops to ~0.037. The proton IS the confinement cliff.',
    keyChannels: 'Color (c[2]=ε) and weak isospin (c[4]=ε) at guard band → IC ≈ 0. This IS confinement.',
  },
  {
    name: 'Neutron', symbol: 'n', scale: 'hadronic', category: 'Baryon',
    c: [0.60, 0.50, 1e-8, 0.33, 1e-8, 1e-8, 0.33, 0.33], w: Array(8).fill(0.125),
    mass: '939.565 MeV/c²', size: '~ 0.86 fm', composition: 'udd — one up quark and two down quarks',
    structure: 'Spin-½ baryon, charge 0, color singlet. 1.293 MeV heavier than proton.',
    stability: 'Free: τ = 879.4 s (14.7 min). Bound in nuclei: stable.',
    signatureNote: 'Even more channels at ε than the proton (charge=0 adds another dead channel). IC/F ≈ 0.009 — the neutron has the most extreme confinement signature of common baryons. Three dead channels produce triple geometric slaughter.',
    keyChannels: 'Color (c[2]=ε), weak isospin (c[4]=ε), stability (c[5]=ε) — three dead channels.',
  },
  {
    name: 'Pion (π⁺)', symbol: 'π⁺', scale: 'hadronic', category: 'Meson',
    c: [0.30, 1.0, 0.0, 0.67, 0.50, 0.50, 0.0, 0.0], w: Array(8).fill(0.125),
    mass: '139.57 MeV/c²', size: '~ 0.66 fm', composition: 'ud̄ — up quark and anti-down quark',
    structure: 'Spin-0 meson (pseudoscalar), charge +1e. Lightest hadron. Mediates nuclear force.',
    stability: 'τ = 26.0 ns — decays to muon + neutrino',
    signatureNote: 'Spin=0 channel (unique among common hadrons). Low mass channel. The pion is the lightest hadron because it is a quasi-Goldstone boson of chiral symmetry breaking.',
    keyChannels: 'Spin (c[2]=0), generation (c[6]=0), stability (c[7]=0) all dead.',
  },

  // ─── NUCLEAR: Nuclei ────────────────────────────────────────────
  {
    name: 'Deuterium', symbol: '²H', scale: 'nuclear', category: 'Light nucleus',
    c: [0.30, 0.95, 0.50, 0.45, 0.70, 0.60, 0.55, 0.80], w: Array(8).fill(0.125),
    mass: '2.014 u (1876 MeV/c²)', size: '~ 2.1 fm', composition: '1 proton + 1 neutron',
    structure: 'Simplest composite nucleus. Binding energy = 2.22 MeV (only 1.1 MeV/nucleon). Spin-1.',
    stability: 'Stable',
    signatureNote: 'Moderate F with no dead channels — all channels contribute. The deuteron sits right at the boundary: barely bound, but all channels are alive. This gives much higher IC than confined hadrons.',
    keyChannels: 'Evenly distributed — no killer channel. Weakest is c[0]=0.30 (Z normalization).',
  },
  {
    name: 'Helium-4', symbol: '⁴He', scale: 'nuclear', category: 'Magic nucleus',
    c: [0.25, 0.95, 0.0, 0.01, 0.98, 0.01, 0.32, 0.0], w: Array(8).fill(0.125),
    mass: '4.0026 u (3727 MeV/c²)', size: '~ 1.67 fm', composition: '2 protons + 2 neutrons (alpha particle)',
    structure: 'Doubly magic nucleus (Z=2, N=2). Extremely tightly bound: BE/A = 7.07 MeV/nucleon.',
    stability: 'Stable — the alpha particle is a nuclear building block',
    signatureNote: 'Very high binding channels but dead electronegativity and electron affinity channels (noble gas has no chemical activity). IC is crushed by the chemistry-dead channels despite excellent nuclear stability.',
    keyChannels: 'c[2]=0 (EN), c[3]=0.01, c[5]=0.01, c[7]=0 — chemistry channels kill IC.',
  },
  {
    name: 'Iron-56', symbol: '⁵⁶Fe', scale: 'nuclear', category: 'Peak binding',
    c: [0.77, 0.82, 0.71, 0.88, 0.78, 0.76, 0.79, 0.65], w: Array(8).fill(0.125),
    mass: '55.845 u', size: '~ 4.1 fm (nuclear) / 140 pm (atomic)',
    composition: '26 protons + 30 neutrons. Electron config: [Ar] 3d⁶ 4s²',
    structure: 'Near peak of nuclear binding curve (8.79 MeV/A). Transition metal, d-block. BCC crystal at room temp.',
    stability: 'Stable — the endpoint of stellar nucleosynthesis in massive stars',
    signatureNote: 'The most balanced kernel signature of any common element. All 8 channels in the 0.65–0.88 range — no dead channels, low spread. This uniformity produces high IC and the smallest heterogeneity gap. Iron IS stability.',
    keyChannels: 'All channels contribute. Weakest is c[7]=0.65 (density-log). Highest is c[3]=0.88.',
  },
  {
    name: 'Lead-208', symbol: '²⁰⁸Pb', scale: 'nuclear', category: 'Doubly magic',
    c: [0.85, 0.55, 0.58, 0.95, 0.42, 0.75, 0.88, 0.82], w: Array(8).fill(0.125),
    mass: '207.97 u', size: '~ 7.1 fm (nuclear) / 175 pm (atomic)',
    composition: '82 protons + 126 neutrons. Both Z and N are magic numbers.',
    structure: 'Doubly magic nucleus — extra stability from closed nuclear shells. Heavy element, FCC crystal.',
    stability: 'Stable — final product of all radioactive decay chains',
    signatureNote: 'Doubly magic gives excellent F from nuclear stability channels. Moderate spread (C~0.15) from the gap between nuclear channels (high) and chemical channels (moderate). IC strong because no channel is dead.',
    keyChannels: 'c[4]=0.42 (IE) is weakest. Nuclear channels (c[0], c[3], c[6]) are strongest.',
  },
  {
    name: 'Uranium-238', symbol: '²³⁸U', scale: 'nuclear', category: 'Actinide',
    c: [0.93, 0.42, 0.55, 0.92, 0.38, 0.82, 0.90, 0.75], w: Array(8).fill(0.125),
    mass: '238.03 u', size: '~ 7.4 fm (nuclear) / 175 pm (atomic)',
    composition: '92 protons + 146 neutrons. Electron config: [Rn] 5f³ 6d¹ 7s²',
    structure: 'Heaviest naturally abundant element. Alpha-emitter. Fissile (²³⁵U). Orthorhombic crystal.',
    stability: 'τ = 4.47 × 10⁹ years (alpha decay)',
    signatureNote: 'High Z gives strong mass and nuclear channels, but chemical channels (EN=0.42, IE=0.38) are depressed. The gap between nuclear strength and chemical weakness creates moderate Δ. Radioactive but cosmologically long-lived.',
    keyChannels: 'c[4]=0.38 (IE) is the weakest — actinides have low ionization energy.',
  },

  // ─── ATOMIC: Elements ───────────────────────────────────────────
  {
    name: 'Hydrogen', symbol: 'H', scale: 'atomic', category: 'Reactive nonmetal',
    c: [0.01, 0.55, 0.93, 0.90, 0.55, 0.21, 0.004, 0.004], w: Array(8).fill(0.125),
    mass: '1.008 u', size: '25 pm (atomic radius)', composition: '1 proton + 1 electron',
    structure: 'Simplest atom. Z=1. Gas at STP. Forms H₂ molecule. The most abundant element in the universe (75% by mass).',
    stability: 'Stable',
    signatureNote: 'Extreme: highest IE/EA per mass, nearly zero density/melting/boiling. Two channels near zero (T_melt, density) destroy IC despite high radius and IE channels. Hydrogen is chemically extreme — its kernel reflects this.',
    keyChannels: 'c[6]=0.004 (T_melt) and c[7]=0.004 (density) near ε — gas with lowest melting point.',
  },
  {
    name: 'Helium', symbol: 'He', scale: 'atomic', category: 'Noble gas',
    c: [0.02, 0.0, 0.91, 0.0, 1.0, 0.0, 0.0008, 0.0003], w: Array(8).fill(0.125),
    mass: '4.003 u', size: '31 pm', composition: '2 protons + 2 neutrons + 2 electrons',
    structure: 'Noble gas. Filled 1s shell (closed shell). Liquid He-4 is a superfluid below 2.17 K. Cannot form chemical bonds.',
    stability: 'Stable — inert',
    signatureNote: 'The most chemically inert element. Zero electronegativity, zero electron affinity — these dead channels annihilate IC. Despite highest ionization energy (c[4]=1.0), the 4 dead channels overwhelm. Helium IS geometric slaughter in the periodic table.',
    keyChannels: 'c[1]=0 (EN), c[3]=0 (EA), c[6]≈0 (T_melt), c[7]≈0 (density) — 4 dead channels.',
  },
  {
    name: 'Carbon', symbol: 'C', scale: 'atomic', category: 'Reactive nonmetal',
    c: [0.05, 0.64, 0.80, 0.82, 0.35, 0.46, 1.00, 0.14], w: Array(8).fill(0.125),
    mass: '12.011 u', size: '70 pm', composition: '6 protons + 6 neutrons + 6 electrons',
    structure: 'Tetravalent. Forms diamond (sp³), graphite (sp²), fullerenes, nanotubes. Basis of organic chemistry. The most versatile bonding element.',
    stability: 'Stable',
    signatureNote: 'Highest melting point of any element (c[6]=1.0 in diamond). Strong covalent bonding channels. Moderate EN and IE. The spread from c[0]=0.05 to c[6]=1.0 creates significant C (curvature), but most channels are active.',
    keyChannels: 'c[6]=1.0 (T_melt — diamond) and c[3]=0.82 (EA) are strongest. c[0]=0.05 (Z) is weakest.',
  },
  {
    name: 'Nitrogen', symbol: 'N', scale: 'atomic', category: 'Reactive nonmetal',
    c: [0.06, 0.77, 0.78, 0.77, 0.0, 0.57, 0.02, 0.02], w: Array(8).fill(0.125),
    mass: '14.007 u', size: '65 pm', composition: '7 protons + 7 neutrons + 7 electrons',
    structure: 'Triple bond in N₂ (945 kJ/mol — strongest homonuclear bond). 78% of atmosphere.',
    stability: 'Stable',
    signatureNote: 'Near-zero electron affinity (c[4]=0) because of half-filled 2p³ shell stability. Also near-zero melting and boiling (diatomic gas). Three channels near ε destroy IC despite moderate EN and IE.',
    keyChannels: 'c[4]=0 (EA), c[6]=0.02 (T_melt), c[7]=0.02 (density) near ε — gas at STP.',
  },
  {
    name: 'Oxygen', symbol: 'O', scale: 'atomic', category: 'Reactive nonmetal',
    c: [0.07, 0.87, 0.81, 0.83, 0.40, 0.54, 0.03, 0.02], w: Array(8).fill(0.125),
    mass: '15.999 u', size: '60 pm', composition: '8 protons + 8 neutrons + 8 electrons',
    structure: 'Diatomic O₂. Paramagnetic. Strong oxidizer. 21% of atmosphere, 46% of Earth\'s crust by mass.',
    stability: 'Stable',
    signatureNote: 'High electronegativity (2nd highest after F) gives strong EN channel. But gas-phase channels (T_melt, density) are near ε. Oxygen is chemically powerful but physically tenuous at STP.',
    keyChannels: 'c[6]=0.03 (T_melt) and c[7]=0.02 (density) — gaseous, kills IC.',
  },
  {
    name: 'Silicon', symbol: 'Si', scale: 'atomic', category: 'Metalloid',
    c: [0.12, 0.45, 0.68, 0.78, 0.36, 0.48, 0.44, 0.17], w: Array(8).fill(0.125),
    mass: '28.086 u', size: '110 pm', composition: '14 protons + 14 neutrons + 14 electrons',
    structure: 'Metalloid, diamond cubic crystal. Band gap 1.12 eV. Basis of semiconductor industry.',
    stability: 'Stable',
    signatureNote: 'Moderate across all channels — no extreme highs or lows. This balanced profile with moderate values makes silicon a "middle-of-the-road" element in kernel space, which correlates with its semiconductor nature (neither conductor nor insulator).',
    keyChannels: 'No dominant channel. c[0]=0.12 (Z) weakest, c[3]=0.78 (EA) strongest.',
  },
  {
    name: 'Iron', symbol: 'Fe', scale: 'atomic', category: 'Transition metal',
    c: [0.22, 0.46, 0.59, 0.85, 0.04, 0.47, 0.61, 0.47], w: Array(8).fill(0.125),
    mass: '55.845 u', size: '140 pm', composition: '26 protons + 30 neutrons + 26 electrons. [Ar] 3d⁶ 4s²',
    structure: 'BCC (α-Fe at room temp). Ferromagnetic. Most abundant element in Earth by mass. Steel alloys dominate construction.',
    stability: 'Stable — stellar nucleosynthesis endpoint for massive stars',
    signatureNote: 'Peak nuclear stability but moderate chemical channels. Electron affinity very low (c[4]=0.04) — iron forms ions easily but doesn\'t attract electrons strongly. The d-block\'s partially filled orbitals give versatile chemistry.',
    keyChannels: 'c[4]=0.04 (EA) is the IC killer — very low electron affinity for a metal.',
  },
  {
    name: 'Copper', symbol: 'Cu', scale: 'atomic', category: 'Transition metal',
    c: [0.25, 0.48, 0.56, 0.87, 0.33, 0.48, 0.56, 0.50], w: Array(8).fill(0.125),
    mass: '63.546 u', size: '128 pm', composition: '29 protons + 35 neutrons + 29 electrons. [Ar] 3d¹⁰ 4s¹',
    structure: 'FCC crystal. Excellent electrical/thermal conductor. Ductile. Forms Cu⁺/Cu²⁺ ions.',
    stability: 'Stable',
    signatureNote: 'Filled d-shell (3d¹⁰) gives excellent conductivity. Balanced channels with no extreme dead zones — moderate IC. Higher EA than iron. The "middle register" of the d-block.',
    keyChannels: 'c[0]=0.25 (Z) weakest. c[3]=0.87 (EA) strongest. Good overall balance.',
  },
  {
    name: 'Gold', symbol: 'Au', scale: 'atomic', category: 'Transition metal',
    c: [0.67, 0.64, 0.58, 0.91, 0.64, 0.38, 0.60, 0.72], w: Array(8).fill(0.125),
    mass: '196.97 u', size: '144 pm', composition: '79 protons + 118 neutrons + 79 electrons. [Xe] 4f¹⁴ 5d¹⁰ 6s¹',
    structure: 'FCC crystal. Highly ductile and malleable. Relativistic effects contract 6s orbital, causing gold color.',
    stability: 'Stable',
    signatureNote: 'High Z gives strong mass channel. Excellent EA (c[3]=0.91) — gold is one of the most electron-greedy metals. Balanced profile with no dead channels. Relativistic electron effects give gold its unique color and chemistry.',
    keyChannels: 'c[5]=0.38 (T_melt) relatively weakest. c[3]=0.91 (EA) dominant — gold loves electrons.',
  },

  // ─── MOLECULAR: Compounds ──────────────────────────────────────
  {
    name: 'Water', symbol: 'H₂O', scale: 'molecular', category: 'Covalent compound',
    c: [0.04, 0.87, 0.83, 0.85, 0.48, 0.55, 0.07, 0.12], w: Array(8).fill(0.125),
    mass: '18.015 u', size: '~ 0.275 nm (O-H bond)', composition: '2 hydrogen + 1 oxygen, covalent bonds',
    structure: 'Bent molecular geometry (104.5°). Strong hydrogen bonding. Anomalous density maximum at 4°C. Universal solvent.',
    stability: 'Stable — but readily participates in reactions',
    signatureNote: 'Water inherits oxygen\'s high EN and EA channels but gains structure from hydrogen bonding. Low molecular mass and gas-phase density keep c[0] and c[7] low. The "bent" structure creates the hydrogen bond network that gives water its anomalous properties.',
    keyChannels: 'c[0]=0.04 (mass) and c[6]=0.07 (T_melt) are lowest — light molecule, low melting point.',
  },
  {
    name: 'Diamond', symbol: 'C (diamond)', scale: 'molecular', category: 'Covalent solid',
    c: [0.05, 0.64, 0.80, 0.82, 0.35, 1.0, 1.0, 0.23], w: Array(8).fill(0.125),
    mass: '12.011 u/atom', size: '0.154 nm C-C bond', composition: 'Pure carbon, sp³ hybridized',
    structure: 'Face-centered cubic (diamond cubic). Each C bonded to 4 others. Hardest natural material. Band gap 5.5 eV (wide gap insulator).',
    stability: 'Metastable (graphite is thermodynamically favored at STP)',
    signatureNote: 'Highest melting/sublimation point (c[5]=1.0, c[6]=1.0) of any element. Pure covalent bonding gives extreme hardness. The kernel sees diamond as "all structure, low mass" — very high bonding channels, very low mass channel.',
    keyChannels: 'c[0]=0.05 (mass) and c[7]=0.23 (density) weakest. c[5]=1.0 and c[6]=1.0 (thermal) max.',
  },
  {
    name: 'Steel (mild)', symbol: 'Fe-C', scale: 'molecular', category: 'Alloy',
    c: [0.23, 0.46, 0.59, 0.84, 0.10, 0.48, 0.58, 0.48], w: Array(8).fill(0.125),
    mass: '~ 55.8 u (avg)', size: 'BCC/FCC grains, 10-100 μm', composition: 'Iron + 0.2% carbon + trace Mn, Si',
    structure: 'Polycrystalline. BCC (ferrite) or FCC (austenite). Martensite on quenching. The most important structural material in civilization.',
    stability: 'Stable (but corrodes without protection)',
    signatureNote: 'Nearly identical to pure Fe kernel — the 0.2% carbon perturbs only the EA and melting channels slightly. Steel demonstrates that small compositional changes barely move the kernel trace, while dramatically changing mechanical properties. The kernel sees bulk composition; microstructure is sub-channel.',
    keyChannels: 'c[4]=0.10 (EA) weakest — same as iron. Alloy additions shift channels < 0.05.',
  },
  {
    name: 'Sodium Chloride', symbol: 'NaCl', scale: 'molecular', category: 'Ionic compound',
    c: [0.15, 0.60, 0.65, 0.98, 0.42, 0.45, 0.21, 0.16], w: Array(8).fill(0.125),
    mass: '58.44 u', size: '0.282 nm (Na-Cl bond)', composition: 'Sodium + chlorine, ionic bond',
    structure: 'FCC rock salt structure. Ionic crystal. Soluble in water. Transparent to visible light.',
    stability: 'Stable — very high lattice energy (786 kJ/mol)',
    signatureNote: 'The EA channel dominates (c[3]=0.98) because Cl has the highest electron affinity of any element. The ionic bond IS the electron transfer from Na to Cl. Low mass and density channels reflect that these are light atoms.',
    keyChannels: 'c[3]=0.98 (EA) dominant — chlorine\'s electron affinity drives the bond. c[0]=0.15 weakest.',
  },

  // ─── BULK: Materials and states ─────────────────────────────────
  {
    name: 'Graphene', symbol: 'C (2D)', scale: 'bulk', category: 'Nanomaterial',
    c: [0.05, 0.64, 0.80, 0.82, 0.35, 0.95, 0.95, 0.03], w: Array(8).fill(0.125),
    mass: '12.011 u/atom', size: '0.335 nm thick (monolayer)', composition: 'Single layer of sp² carbon atoms',
    structure: 'Hexagonal lattice. Zero band gap (semimetal). Strongest material ever measured (130 GPa). Highest electron mobility in any material.',
    stability: 'Stable in vacuum; reactive at edges',
    signatureNote: 'Similar to diamond but with near-zero density channel (2D material). The extreme ratio of bonding strength to mass gives graphene its record properties. Kernel captures this: high bonding channels, near-zero bulk channels.',
    keyChannels: 'c[7]=0.03 (density) near ε — it\'s one atom thick. c[5]=0.95, c[6]=0.95 (thermal) high.',
  },
  {
    name: 'Quartz', symbol: 'SiO₂', scale: 'bulk', category: 'Covalent/Ionic solid',
    c: [0.10, 0.55, 0.72, 0.80, 0.38, 0.55, 0.46, 0.20], w: Array(8).fill(0.125),
    mass: '60.08 u/formula', size: 'SiO₄ tetrahedra, ~ 0.16 nm Si-O', composition: 'Silicon + oxygen, mixed covalent/ionic bonds',
    structure: 'Trigonal crystal (α-quartz). Piezoelectric. SiO₄ tetrahedral framework. Basis of most rocks and glasses.',
    stability: 'Very stable — weathering-resistant',
    signatureNote: 'Moderate across all channels reflecting SiO₂\'s "middle nature" — not extreme in any single property. The tetrahedral SiO₄ network gives mechanical stability (high bonding channels) without extreme thermal properties.',
    keyChannels: 'c[0]=0.10 (mass) and c[7]=0.20 (density) weakest. c[3]=0.80 (EA from O) strongest.',
  },
  {
    name: 'Superconductor (YBCO)', symbol: 'YBa₂Cu₃O₇', scale: 'bulk', category: 'Ceramic superconductor',
    c: [0.45, 0.52, 0.55, 0.70, 0.30, 0.65, 0.55, 0.55], w: Array(8).fill(0.125),
    mass: '666.2 u/formula', size: 'Unit cell ~ 1.17 nm', composition: 'Yttrium + barium + copper + oxygen (perovskite layers)',
    structure: 'Orthorhombic perovskite. Tc = 93 K (above liquid N₂). CuO₂ planes carry supercurrent. Type II superconductor.',
    stability: 'Stable, but brittle ceramic',
    signatureNote: 'Moderately balanced channels — the multi-element composition averages out extremes. The superconducting property comes from CuO₂ plane physics that operates BELOW the kernel\'s channel resolution. The kernel sees the average chemical composition, not the quantum coherence.',
    keyChannels: 'c[4]=0.30 (IE) weakest. No channel dominant — averaging across 4 elements.',
  },

  // ─── ASTRONOMICAL: Stellar objects ──────────────────────────────
  {
    name: 'Sun (Main Sequence)', symbol: '☉', scale: 'astronomical', category: 'G2V star',
    c: [0.92, 0.85, 0.88, 0.90, 0.87, 0.93, 0.86, 0.91], w: Array(8).fill(0.125),
    mass: '1.989 × 10³⁰ kg (1 M☉)', size: '6.96 × 10⁸ m (1 R☉)', composition: '73.5% H, 24.9% He, 1.6% metals (by mass)',
    structure: 'Main sequence star (hydrogen burning). Core: 15.7 × 10⁶ K. Radiative + convective zones. Magnetic dynamo.',
    stability: 'Stable for ~ 5 Gyr more (current age: 4.6 Gyr)',
    signatureNote: 'Extremely balanced: all channels 0.85–0.93. This is what stable hydrogen burning looks like in kernel space — high fidelity, low drift, low curvature. The Sun IS the STABLE regime archetype for astronomy.',
    keyChannels: 'All channels high and uniform → high IC, low Δ. c[1]=0.85 weakest.',
  },
  {
    name: 'Neutron Star', symbol: 'NS', scale: 'astronomical', category: 'Compact object',
    c: [0.95, 0.10, 0.98, 0.05, 0.90, 0.02, 0.88, 0.30], w: Array(8).fill(0.125),
    mass: '1.4 – 2.1 M☉', size: '~ 10 km radius', composition: 'Mostly neutrons + neutron superfluid + possible quark core',
    structure: 'Degenerate matter supported by neutron degeneracy pressure. Crust: nuclear pasta. Core: neutron superfluid. B ~ 10⁸–10¹⁵ T.',
    stability: 'Stable (unless accreting → collapse to black hole)',
    signatureNote: 'Extreme channel asymmetry: density-related channels maximal, temperature channels near zero. The neutron star has the HIGHEST curvature (C) of any astronomical object in the library — maximum channel heterogeneity. This IS what gravitational collapse looks like.',
    keyChannels: 'c[1]=0.10, c[3]=0.05, c[5]=0.02 near floor. c[0]=0.95 and c[2]=0.98 near ceiling.',
  },
  {
    name: 'White Dwarf', symbol: 'WD', scale: 'astronomical', category: 'Compact object',
    c: [0.80, 0.45, 0.92, 0.30, 0.85, 0.15, 0.82, 0.60], w: Array(8).fill(0.125),
    mass: '0.5 – 1.4 M☉', size: '~ 10⁴ km (Earth-sized)', composition: 'C/O core (most common), or He, or O/Ne/Mg',
    structure: 'Electron degenerate matter. Crystallizing core. Chandrasekhar limit: 1.4 M☉. Type Ia supernova progenitor.',
    stability: 'Cooling over 10¹⁰+ years — no fusion',
    signatureNote: 'Intermediate between main-sequence star and neutron star in kernel space. Less extreme channel asymmetry than NS. The crystallization process is visible in the density channel. Watch regime typical — stable but structurally stressed.',
    keyChannels: 'c[1]=0.45 and c[5]=0.15 are weakest. Temperature channels depressed (cooling remnant).',
  },
  {
    name: 'Quark-Gluon Plasma', symbol: 'QGP', scale: 'astronomical', category: 'Extreme state',
    c: [0.35, 0.20, 0.45, 0.30, 0.15, 0.25, 0.40, 0.10], w: Array(8).fill(0.125),
    mass: 'Variable (RHIC: Au+Au at 200 GeV/nucleon)', size: '~ 10 fm (fireball)', composition: 'Deconfined quarks and gluons — primordial soup',
    structure: 'Strongly coupled fluid with near-perfect fluidity (η/s ≈ 1/4π). Color deconfined. Exists for ~10⁻²³ s at RHIC/LHC.',
    stability: 'Extremely transient — hadronizes in ~10⁻²³ s',
    signatureNote: 'ALL channels depressed — no channel above 0.45. This is Collapse regime: high drift, low fidelity, maximum entropy. The QGP signature is "everything partially destroyed simultaneously." The opposite of iron\'s balanced stability.',
    keyChannels: 'c[7]=0.10 (stability) near floor. Everything stressed — collapse regime.',
  },
  {
    name: 'Stellar Black Hole', symbol: 'BH', scale: 'astronomical', category: 'Compact object',
    c: [0.99, 0.50, 0.99, 0.99, 0.60, 0.10, 0.95, 0.40], w: Array(8).fill(0.125),
    mass: '~ 3–20 M☉ (stellar mass)', size: 'r_s = 2GM/c² (~ 9–60 km)', composition: 'Singularity + event horizon — no baryonic structure',
    structure: 'Schwarzschild/Kerr geometry. Event horizon at r_s. Photon sphere at 1.5 r_s. ISCO at 3 r_s (non-rotating). Information paradox: channels collapse at the horizon.',
    stability: 'Eternally stable (classically) — Hawking evaporation on ~ 10⁶⁷ yr timescale',
    signatureNote: 'Extreme heterogeneity: most channels near 1.0 (gravitational dominance) but c[5]=0.10 (information channel) near floor. This IS geometric slaughter — one dead channel kills IC while F stays high. The heterogeneity gap Δ = F − IC is maximal. The BH is the Γ(ω→1) pole made physical.',
    keyChannels: 'c[5]=0.10 (information) kills IC. c[0]=c[2]=c[3]=0.99 (gravitational channels) near ceiling. Maximum Δ.',
  },
  {
    name: 'Event Horizon', symbol: 'EH', scale: 'astronomical', category: 'Spacetime boundary',
    c: [0.10, 0.02, 0.08, 0.02, 0.05, 0.05, 0.45, 0.65], w: Array(8).fill(0.125),
    mass: 'Defined by enclosed mass M', size: 'r_s = 2GM/c²', composition: 'Not a material surface — a causal boundary in spacetime',
    structure: 'Surface of infinite redshift. Outgoing light cones tip inward. In GCD: the ω=1 pole where Γ(ω)→∞. All channels near floor — maximum drift.',
    stability: 'Area theorem: horizon area never decreases (classically)',
    signatureNote: 'Almost all channels at or near floor — this is DEEP collapse. F ≈ 0.18 means 82% of the trace is lost. IC near ε (geometric slaughter across all channels simultaneously). This is the Collapse regime archetype — the physical manifestation of ω approaching 1.0.',
    keyChannels: 'c[1]=c[3]=0.02 at floor. c[6]=0.45  and c[7]=0.65 partially survive (circulation + topology).',
  },
  {
    name: 'Black Hole Photon Sphere', symbol: 'PS', scale: 'astronomical', category: 'Spacetime structure',
    c: [0.30, 0.15, 0.25, 0.12, 0.25, 0.20, 0.55, 0.70], w: Array(8).fill(0.125),
    mass: 'At r = 1.5 r_s (photon orbit)', size: '~ 13.5–90 km (for 3–20 M☉ BH)', composition: 'Unstable circular photon orbits',
    structure: 'GR: null geodesics circle the BH at r = 3GM/c². Unstable — slight perturbation → capture or escape. GCD: intermediate collapse with all channels depressed.',
    stability: 'Unstable equilibrium — perturbations grow exponentially',
    signatureNote: 'All channels in 0.12–0.70 range. Between the event horizon (near floor) and the accretion disk (moderate channels). Watch/Collapse boundary regime. The photon sphere marks where lensing morphology transitions from thick arc to thin arc.',
    keyChannels: 'c[3]=0.12 weakest (tidal channel). c[7]=0.70 strongest (topological persistence).',
  },
  {
    name: 'Accretion Disk', symbol: 'AD', scale: 'astronomical', category: 'Astrophysical structure',
    c: [0.75, 0.60, 0.70, 0.50, 0.65, 0.40, 0.80, 0.55], w: Array(8).fill(0.125),
    mass: '~ 10⁻³ to 10⁻¹ M_BH (mass fraction)', size: '~ 10–10⁵ r_s', composition: 'Ionized gas + dust in Keplerian orbit, angular momentum transport via MRI',
    structure: 'Thin disk (Shakura-Sunyaev) or ADAF. Viscous dissipation converts gravitational energy to radiation. GCD: moderate channels with heterogeneity.',
    stability: 'Quasi-steady state — feeds the BH over 10⁴–10⁸ yr',
    signatureNote: 'Watch regime — intermediate between the stable outer region and the collapse at the horizon. Moderate heterogeneity. F ≈ 0.62, so a third of the signal is lost. The accretion disk is the physical embodiment of the Watch regime in astrophysics.',
    keyChannels: 'c[5]=0.40 weakest (radiative loss channel). c[6]=0.80 strongest (orbital coherence).',
  },
  {
    name: 'Binary BH Merger', symbol: 'BBH', scale: 'astronomical', category: 'Gravitational wave source',
    c: [0.95, 0.85, 0.90, 0.80, 0.90, 0.70, 0.95, 0.95], w: Array(8).fill(0.125),
    mass: '~ 10–150 M☉ (total binary mass)', size: '~ 100–1000 km final BH', composition: 'Two BHs in inspiral + gravitational wave emission',
    structure: 'Three phases: inspiral (PN), merger (NR), ringdown (QNM). GW signal: chirp → peak → damped sinusoid. GCD: high-coherence pre-merger signature.',
    stability: 'Decaying orbit — merger inevitable (GW energy loss)',
    signatureNote: 'Surprisingly high fidelity — the GRAVITATIONAL WAVE SIGNAL preserves information. All channels > 0.70 means the inspiral waveform is coherent. After merger → ringdown → single BH signature (c[5] drops). The binary BH is a RETURN event: information encoded in waves that reach detectors.',
    keyChannels: 'c[5]=0.70 weakest (information carried away by GW). All others ≥ 0.80 — the chirp is clean.',
  },

  // ─── ABSTRACT: Emergent systems ─────────────────────────────────
  {
    name: 'Waking Consciousness', symbol: '—', scale: 'abstract', category: 'Neural coherence',
    c: [0.88, 0.85, 0.82, 0.90, 0.86, 0.84, 0.87, 0.83], w: Array(8).fill(0.125),
    mass: '~ 1.4 kg (brain)', size: '~ 15 cm (brain)', composition: '86 billion neurons, 100 trillion synapses',
    structure: 'Cortical coherence across 8 functional networks. High-frequency (30–100 Hz gamma) binding. Default mode / executive / salience networks.',
    stability: 'Active maintenance required — sleep is mandatory',
    signatureNote: 'Very similar to Sun signature — high fidelity, low curvature, stable regime. Waking consciousness IS the "main sequence" of neural systems: sustained, balanced, high-coherence operation.',
    keyChannels: 'All channels 0.82–0.90. No killer channel. Highest IC of neural states.',
  },
  {
    name: 'S&P 500 (Bull Market)', symbol: '—', scale: 'abstract', category: 'Market coherence',
    c: [0.92, 0.88, 0.85, 0.90, 0.87, 0.93, 0.80, 0.91], w: Array(8).fill(0.125),
    mass: '~ $40 trillion market cap', size: 'Global (500 companies)', composition: '500 large-cap US equities, market-weighted',
    structure: 'Index of 11 GICS sectors. Bull market: sustained appreciation with low volatility and positive breadth.',
    stability: 'Regime-dependent — bull markets last 3-10 years typically',
    signatureNote: 'Bull market signature closely resembles the Sun and Waking Consciousness — high uniform channels. Markets in bull phase achieve "stellar stability" in kernel space. The lowest channel (c[6]=0.80) represents lagging sectors.',
    keyChannels: 'c[6]=0.80 weakest (sector lag). All others > 0.85 — broad market health.',
  },
  {
    name: 'Black Monday (1987)', symbol: '—', scale: 'abstract', category: 'Market collapse',
    c: [0.15, 0.10, 0.20, 0.05, 0.30, 0.12, 0.08, 0.25], w: Array(8).fill(0.125),
    mass: '~ $500 billion lost in one day', size: 'Global contagion', composition: 'Cascading sell orders, portfolio insurance failure',
    structure: 'Dow lost 22.6% in one day. The largest single-day percentage drop in history. No single cause — systemic.',
    stability: 'Recovered within 2 years',
    signatureNote: 'Almost identical to QGP signature — total collapse across all channels simultaneously. Market crash IS the financial analog of deconfinement: all coherence structures fail at once. Collapse regime, deep critical overlay.',
    keyChannels: 'c[3]=0.05 weakest (liquidity channel). Everything below 0.30 except c[4]=0.30.',
  },
];

/* ─── Kernel computation for entities (cached) ──────────────────── */

function getEntityKernel(entity: PhysicalEntity): KernelResult {
  if (!entity._kernel) {
    entity._kernel = computeKernel(entity.c, entity.w);
  }
  return entity._kernel;
}

/* ─── Signature Distance ────────────────────────────────────────── */

export interface SignatureMatch {
  entity: PhysicalEntity;
  distance: number;       // Euclidean distance in kernel space
  kernelResult: KernelResult;
  regime: RegimeLabel;
  isCritical: boolean;
  matchQuality: 'exact' | 'close' | 'similar' | 'approximate' | 'distant';
}

/**
 * Compute weighted Euclidean distance between two kernel results.
 * Weights emphasize the most structurally meaningful invariants:
 *   F, IC, Δ get weight 2 (primary structural identity)
 *   S, C, ω get weight 1 (secondary)
 */
export function kernelDistance(a: KernelResult, b: KernelResult): number {
  const dF = (a.F - b.F) * 2;
  const dIC = (a.IC - b.IC) * 2;
  const dDelta = (a.delta - b.delta) * 2;
  const dS = a.S - b.S;
  const dC = a.C - b.C;
  const dOmega = a.omega - b.omega;
  return Math.sqrt(dF * dF + dIC * dIC + dDelta * dDelta + dS * dS + dC * dC + dOmega * dOmega);
}

/**
 * Find the N closest physical entities to a given kernel result.
 */
export function findNearestEntities(
  target: KernelResult,
  n: number = 10,
  scaleFilter?: PhysicalScale,
): SignatureMatch[] {
  const candidates = scaleFilter
    ? PHYSICAL_ENTITIES.filter(e => e.scale === scaleFilter)
    : PHYSICAL_ENTITIES;

  const matches: SignatureMatch[] = candidates.map(entity => {
    const kr = getEntityKernel(entity);
    const regime = classifyRegime(kr);
    const dist = kernelDistance(target, kr);
    let matchQuality: SignatureMatch['matchQuality'];
    if (dist < 0.01) matchQuality = 'exact';
    else if (dist < 0.05) matchQuality = 'close';
    else if (dist < 0.15) matchQuality = 'similar';
    else if (dist < 0.35) matchQuality = 'approximate';
    else matchQuality = 'distant';

    return {
      entity, distance: dist, kernelResult: kr,
      regime: regime.regime, isCritical: regime.isCritical, matchQuality,
    };
  });

  matches.sort((a, b) => a.distance - b.distance);
  return matches.slice(0, n);
}

/* ─── Scale Classifier ──────────────────────────────────────────── */

export interface ScaleClassification {
  predictedScale: PhysicalScale;
  confidence: number;     // 0–1 (based on distance to nearest in that scale)
  scaleScores: Record<PhysicalScale, number>;
  reasoning: string;
}

/**
 * Classify which physical scale a kernel result most likely belongs to.
 * Uses k-nearest-neighbors voting weighted by inverse distance.
 */
export function classifyScale(target: KernelResult, k: number = 5): ScaleClassification {
  const nearest = findNearestEntities(target, k);
  const scores: Record<PhysicalScale, number> = {
    subatomic: 0, hadronic: 0, nuclear: 0, atomic: 0,
    molecular: 0, bulk: 0, astronomical: 0, abstract: 0,
  };

  for (const match of nearest) {
    const weight = 1 / (match.distance + 0.001);
    scores[match.entity.scale] += weight;
  }

  // Normalize
  const total = Object.values(scores).reduce((s, v) => s + v, 0);
  for (const key of Object.keys(scores) as PhysicalScale[]) {
    scores[key] /= total || 1;
  }

  // Find winner
  let best: PhysicalScale = 'atomic';
  let bestScore = 0;
  for (const [scale, score] of Object.entries(scores) as [PhysicalScale, number][]) {
    if (score > bestScore) { bestScore = score; best = scale; }
  }

  // Generate reasoning
  const top3 = nearest.slice(0, 3);
  const reasoning = `Nearest: ${top3.map(m => `${m.entity.name} (${m.entity.scale}, d=${m.distance.toFixed(3)})`).join(', ')}. ` +
    `Scale vote: ${best} at ${(bestScore * 100).toFixed(0)}% confidence.`;

  return { predictedScale: best, confidence: bestScore, scaleScores: scores, reasoning };
}

/* ─── Physical Interpretation ───────────────────────────────────── */

export interface PhysicalInterpretation {
  summary: string;
  scaleVerdict: string;
  structureVerdict: string;
  compositionHints: string[];
  stabilityVerdict: string;
  anomalies: string[];
  nearestMatch: string;
  noveltyScore: number;   // 0 = exact match, 1 = nothing similar
}

/**
 * Generate a human-readable physical interpretation of a kernel result.
 * This is the core "what does this number mean?" function.
 */
export function interpretKernel(
  result: KernelResult,
  channels?: number[],
): PhysicalInterpretation {
  const regime = classifyRegime(result);
  const nearest = findNearestEntities(result, 5);
  const scaleClass = classifyScale(result);
  const noveltyScore = nearest[0]?.distance ?? 1.0;

  // Summary
  const buildSummary = (): string => {
    if (regime.regime === 'STABLE' && !regime.isCritical) {
      return `High-coherence system (F=${result.F.toFixed(3)}, IC=${result.IC.toFixed(3)}). Structurally similar to stable matter or sustained processes. All invariants within stable thresholds.`;
    }
    if (regime.regime === 'COLLAPSE') {
      return `Low-coherence system (F=${result.F.toFixed(3)}, ω=${result.omega.toFixed(3)}). Structurally similar to deconfined or disrupted states. Significant channel degradation across the trace.`;
    }
    return `Intermediate-coherence system (F=${result.F.toFixed(3)}, IC=${result.IC.toFixed(3)}). Watch regime — operational but with structural stress. ${result.delta > 0.1 ? 'Significant heterogeneity gap (Δ=' + result.delta.toFixed(3) + ').' : 'Moderate heterogeneity.'}`;
  };

  // Structure verdict
  const buildStructure = (): string => {
    if (result.delta < 0.01) return 'Homogeneous: all channels contribute equally. Like a pure element or uniform medium.';
    if (result.delta < 0.05) return 'Nearly homogeneous: minor channel variation. Like a well-mixed alloy with slight compositional gradient.';
    if (result.delta < 0.15) return 'Mildly heterogeneous: some channels stronger than others. Like a compound with mixed bond types.';
    if (result.delta < 0.30) return 'Significantly heterogeneous: large gap between strongest and weakest channels. Like a composite material or system with one dominant property.';
    return 'Severely heterogeneous (geometric slaughter): at least one channel near ε destroys multiplicative coherence. Like confinement (dead color channel) or noble gases (dead chemistry channels).';
  };

  // Composition hints from channel patterns
  const buildCompositionHints = (): string[] => {
    const hints: string[] = [];
    if (!channels || channels.length === 0) {
      if (result.F > 0.90 && result.C < 0.05) hints.push('Uniform composition — single-element or pure-phase material');
      if (result.F > 0.85 && result.C > 0.10) hints.push('Mixed composition with one dominant component');
      if (result.IC < 0.01) hints.push('At least one critical property is near zero — confined, inert, or fundamentally absent channel');
      if (result.delta > 0.40) hints.push('Extreme structural asymmetry — some properties maximal while others absent (e.g., confinement, noble gas inertness)');
      return hints;
    }

    const minC = Math.min(...channels);
    const maxC = Math.max(...channels);
    const nDead = channels.filter(c => c < 0.05).length;
    const nStrong = channels.filter(c => c > 0.85).length;

    if (nDead >= 3) hints.push(`${nDead} dead channels (< 0.05) — sparse signature, like a fundamental particle or inert gas`);
    else if (nDead >= 1) hints.push(`${nDead} dead channel(s) — geometric slaughter active, IC severely depressed`);

    if (nStrong >= 6) hints.push(`${nStrong} strong channels (> 0.85) — uniform high-fidelity, like stable bulk matter or main-sequence processes`);
    if (maxC > 0.95 && minC < 0.10) hints.push('Maximum spread: some channels at ceiling, others at floor — extreme anisotropy');

    if (result.F > 0.75 && result.IC < 0.10) hints.push('High F but crushed IC: textbook geometric slaughter — one channel poisons the geometric mean');
    if (result.S > 0.50) hints.push('High Bernoulli entropy — many channels near 0.5 (maximum uncertainty per channel)');
    if (result.C > 0.30) hints.push('High curvature — extreme channel spread, likely heterogeneous composition or mixed-phase system');

    return hints;
  };

  // Stability verdict
  const buildStability = (): string => {
    if (regime.regime === 'STABLE' && !regime.isCritical && result.delta < 0.05) {
      return 'Highly stable: low drift, high coherence, balanced channels. Physically corresponds to equilibrium states, stable nuclei, main-sequence stars, or healthy systems.';
    }
    if (regime.regime === 'STABLE') {
      return 'Stable but with structural stress. All gates pass, but the heterogeneity gap suggests compositional non-uniformity.';
    }
    if (regime.regime === 'WATCH' && !regime.isCritical) {
      return 'Metastable: system is operational but drifting. Could correspond to radioactive elements, market volatility, or transitional phases.';
    }
    if (regime.isCritical) {
      return 'Critical: IC < 0.30 means multiplicative coherence is dangerously low. At least one channel is near collapse. Physically: unstable particles, deconfined matter, or system failure.';
    }
    return 'Collapse regime: significant structural degradation. Physically: transient states, disrupted systems, or extreme conditions.';
  };

  // Anomaly detection
  const buildAnomalies = (): string[] => {
    const anomalies: string[] = [];
    if (noveltyScore > 0.30) {
      anomalies.push(`No close match in reference library (nearest: d=${noveltyScore.toFixed(3)}). This kernel signature may represent an unknown or novel structure.`);
    }
    if (result.IC > result.F + 1e-10) {
      anomalies.push('WARNING: IC > F — integrity bound violated. Check channel clamping or computation.');
    }
    if (result.F > 0.80 && result.IC < 0.05) {
      anomalies.push('Extreme geometric slaughter: F is healthy but IC is near zero. Consistent with confined systems (hadrons) or noble gases.');
    }
    if (result.C > 0.40) {
      anomalies.push('Extremely high curvature — channels are maximally spread. This is rare and suggests a system with contradictory properties at extreme parameter values.');
    }
    if (regime.regime === 'STABLE' && result.delta > 0.15) {
      anomalies.push('Stable regime but large heterogeneity gap — anomalous combination. Usually stable systems have Δ < 0.05.');
    }
    return anomalies;
  };

  return {
    summary: buildSummary(),
    scaleVerdict: `Predicted scale: **${SCALE_INFO[scaleClass.predictedScale].label}** (${(scaleClass.confidence * 100).toFixed(0)}% confidence). ${SCALE_INFO[scaleClass.predictedScale].description}. Size range: ${SCALE_INFO[scaleClass.predictedScale].sizeRange}.`,
    structureVerdict: buildStructure(),
    compositionHints: buildCompositionHints(),
    stabilityVerdict: buildStability(),
    anomalies: buildAnomalies(),
    nearestMatch: nearest[0]
      ? `${nearest[0].entity.name} (${nearest[0].entity.scale}, d=${nearest[0].distance.toFixed(4)}): ${nearest[0].entity.signatureNote.slice(0, 120)}...`
      : 'No reference entities available.',
    noveltyScore: Math.min(1.0, noveltyScore / 0.50),  // Normalize: 0.5 distance → 100% novel
  };
}

/* ─── Predicted Properties (from nearest-neighbor regression) ───── */

export interface PredictedProperties {
  massRange: string;
  sizeRange: string;
  likelyComposition: string;
  likelyStructure: string;
  stabilityEstimate: string;
  predictedScale: PhysicalScale;
  confidence: number;
  basis: string[];    // Which entities informed the prediction
}

/**
 * Predict physical properties for an unknown kernel result
 * using inverse-distance-weighted nearest neighbor regression.
 */
export function predictProperties(target: KernelResult): PredictedProperties {
  const nearest = findNearestEntities(target, 5);
  const scaleClass = classifyScale(target);

  if (nearest.length === 0) {
    return {
      massRange: 'Unknown', sizeRange: 'Unknown', likelyComposition: 'Unknown',
      likelyStructure: 'Unknown', stabilityEstimate: 'Unknown',
      predictedScale: 'atomic', confidence: 0, basis: [],
    };
  }

  // Weighted vote for composition and structure
  const top3 = nearest.slice(0, 3);
  const totalWeight = top3.reduce((s, m) => s + 1 / (m.distance + 0.001), 0);

  // If best match is very close, just use it directly
  if (nearest[0].distance < 0.05) {
    const e = nearest[0].entity;
    return {
      massRange: e.mass,
      sizeRange: e.size,
      likelyComposition: e.composition,
      likelyStructure: e.structure,
      stabilityEstimate: e.stability,
      predictedScale: e.scale,
      confidence: Math.max(0, 1 - nearest[0].distance * 10),
      basis: [e.name],
    };
  }

  // Otherwise, synthesize from top matches
  const scales = top3.map(m => m.entity.scale);
  const dominantScale = scaleClass.predictedScale;

  return {
    massRange: `Between ${top3[0].entity.mass} and ${top3[top3.length - 1].entity.mass}`,
    sizeRange: SCALE_INFO[dominantScale].sizeRange,
    likelyComposition: `Hybrid of: ${top3.map(m => m.entity.composition.split('.')[0].split(',')[0]).join(' + ')}`,
    likelyStructure: `Structural blend: ${top3.map(m => m.entity.structure.split('.')[0]).join(' / ')}`,
    stabilityEstimate: top3[0].entity.stability,
    predictedScale: dominantScale,
    confidence: scaleClass.confidence,
    basis: top3.map(m => `${m.entity.name} (d=${m.distance.toFixed(3)})`),
  };
}
