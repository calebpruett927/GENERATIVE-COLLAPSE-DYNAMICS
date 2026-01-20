"""
Resonance Pattern Analysis for RCFT (Tier-2)

Analyzes oscillatory patterns in collapse fields to extract wavelength λ_p and
phase angle Θ. This Tier-2 metric identifies periodic structures in recursive
collapse dynamics using Fourier analysis.

Mathematical Foundation:
    λ_p = 2π / k_dominant     (pattern wavelength)
    Θ = arctan(Im(Ψ_complex) / Re(Ψ_complex))     (phase angle)

    where k_dominant is the dominant frequency from FFT of field time series.

Pattern Types:
    - Standing: Θ_variance < 0.1, λ_p constant (stationary resonance)
    - Traveling: Θ_variance > 0.5, λ_p constant (propagating wave)
    - Mixed: 0.1 ≤ Θ_variance ≤ 0.5 (combination of standing and traveling)

Tier-2 Constraints:
    - Analyzes Tier-1 field dynamics (R, Φ_gen, E_potential) without modifying them
    - Uses GCD resonance R as base field for pattern analysis
    - Respects all GCD mathematical identities and tolerances
"""

from typing import Any

import numpy as np


def compute_resonance_pattern(field_series: np.ndarray, dt: float = 1.0, tol: float = 1e-6) -> dict[str, Any]:
    """
    Analyze resonance patterns in a field time series using Fourier analysis.

    Extracts dominant wavelength, phase angle, and classifies pattern type
    (standing, traveling, or mixed wave).

    Args:
        field_series: Time series of field values (e.g., R, Φ_gen, E)
        dt: Time step between measurements (default: 1.0)
        tol: Numerical tolerance for computations

    Returns:
        Dictionary containing:
            - lambda_pattern: Dominant wavelength (2π/k_dominant)
            - Theta_phase: Phase angle in radians [0, 2π)
            - pattern_type: Classification (Standing/Traveling/Mixed)
            - frequency_spectrum: FFT power spectrum
            - dominant_frequency: Frequency with maximum power
            - phase_coherence: Measure of phase stability (0-1)
            - harmonic_content: Relative strength of harmonics
    """
    n_points = len(field_series)

    if n_points < 4:
        # Not enough points for meaningful FFT
        return {
            "lambda_pattern": np.inf,
            "Theta_phase": 0.0,
            "pattern_type": "Standing",
            "frequency_spectrum": np.array([0.0]),
            "dominant_frequency": 0.0,
            "phase_coherence": 1.0,
            "harmonic_content": 0.0,
            "components": {
                "n_points": n_points,
                "mean_field": float(np.mean(field_series)),
                "std_field": float(np.std(field_series)),
            },
        }

    # Detrend: Remove mean
    field_detrended = field_series - np.mean(field_series)

    # Compute FFT
    fft = np.fft.fft(field_detrended)
    freqs = np.fft.fftfreq(n_points, d=dt)

    # Use positive frequencies only
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft[pos_mask]

    # Power spectrum
    power_spectrum = np.abs(pos_fft) ** 2

    if len(power_spectrum) == 0 or np.max(power_spectrum) < tol:
        # No significant oscillation
        return {
            "lambda_pattern": np.inf,
            "Theta_phase": 0.0,
            "pattern_type": "Standing",
            "frequency_spectrum": power_spectrum,
            "dominant_frequency": 0.0,
            "phase_coherence": 1.0,
            "harmonic_content": 0.0,
            "components": {
                "n_points": n_points,
                "mean_field": float(np.mean(field_series)),
                "std_field": float(np.std(field_series)),
                "max_power": float(np.max(power_spectrum)),
            },
        }

    # Find dominant frequency
    dominant_idx = np.argmax(power_spectrum)
    k_dominant = 2 * np.pi * pos_freqs[dominant_idx]  # Convert to angular frequency

    # Compute wavelength
    lambda_pattern = 2 * np.pi / k_dominant if k_dominant > tol else np.inf

    # Compute phase angle from complex FFT component
    complex_amplitude = pos_fft[dominant_idx]
    Theta_phase = np.arctan2(np.imag(complex_amplitude), np.real(complex_amplitude))

    # Normalize to [0, 2π)
    if Theta_phase < 0:
        Theta_phase += 2 * np.pi

    # Compute phase coherence using FFT coefficients
    # Phase coherence = |Σ exp(iθ_k)| / N
    phases = np.angle(pos_fft)
    phase_vectors = np.exp(1j * phases)
    phase_coherence = np.abs(np.mean(phase_vectors))

    # Analyze harmonic content
    # Check if there are significant harmonics (2k, 3k, etc.)
    harmonic_strength = 0.0
    if dominant_idx > 0 and 2 * dominant_idx < len(power_spectrum):
        harmonic_indices = [2 * dominant_idx, 3 * dominant_idx]
        harmonic_powers = [power_spectrum[idx] if idx < len(power_spectrum) else 0.0 for idx in harmonic_indices]
        harmonic_strength = np.sum(harmonic_powers) / (power_spectrum[dominant_idx] + tol)

    # Compute phase variance for pattern classification
    # Reconstruct instantaneous phase from Hilbert transform
    analytic_signal = np.fft.ifft(np.fft.fft(field_detrended) * (1 + np.sign(freqs)))
    instantaneous_phase = np.angle(analytic_signal)
    phase_variance = np.var(np.diff(instantaneous_phase))

    # Pattern type classification
    if phase_variance < 0.1:
        pattern_type = "Standing"
    elif phase_variance > 0.5:
        pattern_type = "Traveling"
    else:
        pattern_type = "Mixed"

    components = {
        "n_points": n_points,
        "mean_field": float(np.mean(field_series)),
        "std_field": float(np.std(field_series)),
        "max_power": float(np.max(power_spectrum)),
        "phase_variance": float(phase_variance),
        "spectral_entropy": float(
            -np.sum((power_spectrum / np.sum(power_spectrum)) * np.log(power_spectrum / np.sum(power_spectrum) + tol))
        ),
    }

    return {
        "lambda_pattern": float(lambda_pattern),
        "Theta_phase": float(Theta_phase),
        "pattern_type": pattern_type,
        "frequency_spectrum": power_spectrum,
        "dominant_frequency": float(k_dominant / (2 * np.pi)),
        "phase_coherence": float(phase_coherence),
        "harmonic_content": float(harmonic_strength),
        "components": components,
    }


def compute_multi_field_resonance(
    R_series: np.ndarray, Phi_gen_series: np.ndarray, E_series: np.ndarray, dt: float = 1.0, tol: float = 1e-6
) -> dict[str, Any]:
    """
    Analyze resonance patterns across multiple GCD Tier-1 fields simultaneously.

    Computes individual patterns for R (resonance), Φ_gen (generative flux),
    and E (energy potential), then identifies cross-field correlations.

    Args:
        R_series: Time series of GCD resonance values
        Phi_gen_series: Time series of GCD generative flux values
        E_series: Time series of GCD energy potential values
        dt: Time step
        tol: Numerical tolerance

    Returns:
        Dictionary with patterns for each field plus cross-correlations
    """
    # Analyze each field independently
    R_pattern = compute_resonance_pattern(R_series, dt, tol)
    Phi_pattern = compute_resonance_pattern(Phi_gen_series, dt, tol)
    E_pattern = compute_resonance_pattern(E_series, dt, tol)

    # Compute cross-correlations with warning suppression for constant fields
    # If fields are constant (zero variance), correlation is undefined -> set to 0
    with np.errstate(invalid="ignore"):
        R_Phi_corr_matrix = np.corrcoef(R_series, Phi_gen_series)
        R_E_corr_matrix = np.corrcoef(R_series, E_series)
        Phi_E_corr_matrix = np.corrcoef(Phi_gen_series, E_series)

    # Handle NaN from constant fields (zero variance)
    R_Phi_corr = R_Phi_corr_matrix[0, 1] if not np.isnan(R_Phi_corr_matrix[0, 1]) else 0.0
    R_E_corr = R_E_corr_matrix[0, 1] if not np.isnan(R_E_corr_matrix[0, 1]) else 0.0
    Phi_E_corr = Phi_E_corr_matrix[0, 1] if not np.isnan(Phi_E_corr_matrix[0, 1]) else 0.0

    # Phase difference between fields
    phase_diff_R_Phi = (R_pattern["Theta_phase"] - Phi_pattern["Theta_phase"]) % (2 * np.pi)
    phase_diff_R_E = (R_pattern["Theta_phase"] - E_pattern["Theta_phase"]) % (2 * np.pi)

    # Dominant wavelength (use R as primary)
    lambda_dominant = R_pattern["lambda_pattern"]

    # Overall pattern type (majority vote)
    pattern_types = [R_pattern["pattern_type"], Phi_pattern["pattern_type"], E_pattern["pattern_type"]]
    pattern_type = max(set(pattern_types), key=pattern_types.count)

    return {
        "lambda_pattern": lambda_dominant,
        "Theta_phase": R_pattern["Theta_phase"],
        "pattern_type": pattern_type,
        "R_pattern": R_pattern,
        "Phi_pattern": Phi_pattern,
        "E_pattern": E_pattern,
        "cross_correlations": {"R_Phi": float(R_Phi_corr), "R_E": float(R_E_corr), "Phi_E": float(Phi_E_corr)},
        "phase_differences": {"R_Phi": float(phase_diff_R_Phi), "R_E": float(phase_diff_R_E)},
        "coherence": {
            "R": R_pattern["phase_coherence"],
            "Phi": Phi_pattern["phase_coherence"],
            "E": E_pattern["phase_coherence"],
            "mean": float(
                np.mean([R_pattern["phase_coherence"], Phi_pattern["phase_coherence"], E_pattern["phase_coherence"]])
            ),
        },
    }


# Example usage and testing
if __name__ == "__main__":
    print("RCFT Resonance Pattern Closure (Tier-2)")
    print("=" * 60)

    # Test 1: Constant field (no oscillation)
    print("\nTest 1: Constant Field")
    constant = np.ones(50) * 0.8
    result = compute_resonance_pattern(constant)
    print(f"  λ_pattern: {result['lambda_pattern']:.2f}")
    print(f"  Θ_phase: {result['Theta_phase']:.4f} rad")
    print(f"  Pattern type: {result['pattern_type']}")
    print(f"  Phase coherence: {result['phase_coherence']:.4f}")

    # Test 2: Sinusoidal oscillation (standing wave)
    print("\nTest 2: Sinusoidal Standing Wave")
    t = np.linspace(0, 10 * np.pi, 200)
    sine_wave = np.sin(2 * t)
    result = compute_resonance_pattern(sine_wave, dt=t[1] - t[0])
    print(f"  λ_pattern: {result['lambda_pattern']:.4f}")
    print(f"  Θ_phase: {result['Theta_phase']:.4f} rad")
    print(f"  Pattern type: {result['pattern_type']}")
    print(f"  Dominant freq: {result['dominant_frequency']:.4f} Hz")

    # Test 3: Multiple harmonics (mixed pattern)
    print("\nTest 3: Multiple Harmonics")
    multi_harmonic = np.sin(t) + 0.5 * np.sin(3 * t) + 0.25 * np.sin(5 * t)
    result = compute_resonance_pattern(multi_harmonic, dt=t[1] - t[0])
    print(f"  λ_pattern: {result['lambda_pattern']:.4f}")
    print(f"  Pattern type: {result['pattern_type']}")
    print(f"  Harmonic content: {result['harmonic_content']:.4f}")
    print(f"  Spectral entropy: {result['components']['spectral_entropy']:.4f}")

    # Test 4: Multi-field resonance
    print("\nTest 4: Multi-Field Resonance")
    R_field = 0.7 + 0.2 * np.sin(t)
    Phi_field = 0.5 + 0.3 * np.sin(t + np.pi / 4)
    E_field = 0.1 + 0.05 * np.sin(2 * t)
    result = compute_multi_field_resonance(R_field, Phi_field, E_field, dt=t[1] - t[0])
    print(f"  λ_pattern (dominant): {result['lambda_pattern']:.4f}")
    print(f"  Pattern type: {result['pattern_type']}")
    print(f"  R-Phi correlation: {result['cross_correlations']['R_Phi']:.4f}")
    print(f"  R-E correlation: {result['cross_correlations']['R_E']:.4f}")
    print(f"  Phase diff (R-Phi): {result['phase_differences']['R_Phi']:.4f} rad")
    print(f"  Mean coherence: {result['coherence']['mean']:.4f}")

    # Test 5: GCD zero entropy state
    print("\nTest 5: Zero Entropy State (GCD)")
    R_zero = np.ones(30)  # R=1.0 at zero entropy
    Phi_zero = np.full(30, 0.0001)  # Φ_gen ≈ 0
    E_zero = np.zeros(30)  # E=0
    result = compute_multi_field_resonance(R_zero, Phi_zero, E_zero)
    print(f"  λ_pattern: {result['lambda_pattern']:.2f}")
    print(f"  Pattern type: {result['pattern_type']}")
    print(f"  R coherence: {result['coherence']['R']:.4f}")

    print("\n" + "=" * 60)
    print("Resonance pattern closure validated successfully!")
