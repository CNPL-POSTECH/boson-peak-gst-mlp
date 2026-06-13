#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit

from dynasor import read_sample_from_npz
from dynasor.post_processing import compute_spherical_qpoint_average


# ============================================================
# SETTINGS
# ============================================================
NPZ_FILE = "test.npz"
Q_BINS = 60

# time window for fitting (fs)
TMAX_FIT_FS = 5000.0

# frequency window for initial guess (rad/fs)
WMIN_GUESS = 0.005
WMAX_GUESS = 0.25

OUT_TXT = "DHO_dispersion_Gamma_vs_q.txt"

# ---- unit conversion (version independent) ----
# If dynasor omega is in rad/fs, then:
#   E(meV) = ħ * ω = 658.2119569 * ω(rad/fs)
RAD_PER_FS_TO_MEV = 658.2119569


# ============================================================
# DHO model for velocity/current autocorrelation
# ============================================================
def acf_velocity_dho(t, omega0, gamma, A):
    """
    Velocity-ACF form of a damped harmonic oscillator.

    y(t) = A * exp(-gamma t / 2) *
           [ cos(omega_d t) - (gamma / (2 omega_d)) sin(omega_d t) ]

    omega_d = sqrt(omega0^2 - (gamma/2)^2)

    Units:
      t      : fs
      omega0 : rad/fs
      gamma  : rad/fs
    """
    t = np.asarray(t)
    inside = omega0**2 - (gamma / 2.0)**2
    omega_d = np.sqrt(np.maximum(inside, 1e-16))

    return A * np.exp(-gamma * t / 2.0) * (
        np.cos(omega_d * t)
        - (gamma / (2.0 * omega_d)) * np.sin(omega_d * t)
    )


# ============================================================
# Helper functions
# ============================================================
def find_peak_omega(omega_rad, spectrum):
    """Find peak position in frequency domain for initial guess."""
    mask = (
        (omega_rad > WMIN_GUESS)
        & (omega_rad < WMAX_GUESS)
        & np.isfinite(spectrum)
    )
    if not np.any(mask):
        return None
    return omega_rad[mask][np.argmax(spectrum[mask])]


def fit_dho_one_q(t_fs, y_t, omega_guess):
    """Fit DHO to one q trace."""
    mask = (t_fs >= 0) & (t_fs <= TMAX_FIT_FS) & np.isfinite(y_t)
    tfit = t_fs[mask]
    yfit = y_t[mask]

    if len(tfit) < 10:
        raise RuntimeError("Not enough points for fitting")

    if omega_guess is None:
        omega_guess = 0.05  # fallback (rad/fs)

    A0 = yfit[0] if abs(yfit[0]) > 1e-12 else 1.0
    gamma0 = max(omega_guess * 0.2, 1e-4)

    p0 = [omega_guess, gamma0, A0]
    bounds = ([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(
        acf_velocity_dho,
        tfit,
        yfit,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    return popt  # omega0, gamma, A


# ============================================================
# Main
# ============================================================
def main():
    # Load dynasor data
    sample_raw = read_sample_from_npz(NPZ_FILE)
    sample_averaged = compute_spherical_qpoint_average(
        sample_raw, q_bins=Q_BINS
    )

    # IMPORTANT: omega must be in rad/fs for fitting (do NOT convert beforehand)
    t_fs = sample_averaged.time
    omega_rad = sample_averaged.omega
    q_vals = sample_averaged.q_norms

    Clqt = sample_averaged.Clqt
    Ctqt = sample_averaged.Ctqt
    Clqw = sample_averaged.Clqw
    Ctqw = sample_averaged.Ctqw

    rows = []

    for iq, q in enumerate(q_vals):
        try:
            # initial guesses from frequency domain
            wL_guess = find_peak_omega(omega_rad, Clqw[iq])
            wT_guess = find_peak_omega(omega_rad, Ctqw[iq])

            # time-domain DHO fitting
            w0L, gL, _ = fit_dho_one_q(t_fs, Clqt[iq], wL_guess)
            w0T, gT, _ = fit_dho_one_q(t_fs, Ctqt[iq], wT_guess)

            rows.append([
                iq,
                q,
                w0L * RAD_PER_FS_TO_MEV,
                gL  * RAD_PER_FS_TO_MEV,
                w0T * RAD_PER_FS_TO_MEV,
                gT  * RAD_PER_FS_TO_MEV,
            ])

        except Exception as e:
            print(f"[WARN] q-index {iq} fit failed: {e}")

    if len(rows) == 0:
        raise RuntimeError("No successful fits")

    rows = np.array(rows)

    header = (
        "q_index  q(1/Ang)  "
        "omega_L(meV)  gamma_L(meV)  "
        "omega_T(meV)  gamma_T(meV)"
    )

    np.savetxt(
        OUT_TXT,
        rows,
        header=header,
        fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f"],
    )

    print(f"Saved: {OUT_TXT}")
    print("NOTE: conversion uses 1 rad/fs = 658.2119569 meV")


if __name__ == "__main__":
    main()

