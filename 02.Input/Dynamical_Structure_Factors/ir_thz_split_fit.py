#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ============================================================
# USER SETTINGS
# ============================================================
INPUT_TXT = "DHO_dispersion_Gamma_vs_q.txt"

OUT_ALL = "IR_L_T_THz_fit.png"
OUT_L   = "IR_L_only_THz_fit.png"
OUT_T   = "IR_T_only_THz_fit.png"

# ---------- Axis control ----------
# y-axis (THz)
Y_MIN = 0.0
Y_MAX = 2       
Y_MAJOR = 1.0
Y_MINOR = 0.25

# x-axis (q, 1/Å)
X_MIN = None       
X_MAX = 0.5        

# ---------- IR search range (optional) ----------
QMIN_IR = None
QMAX_IR = None

# ---------- Fit q-window (ONLY this range is used for fitting) ----------
# Omega(q) ~ v*q + b  : fit only in [FIT_QMIN_OMEGA, FIT_QMAX_OMEGA]
FIT_QMIN_OMEGA = None     
FIT_QMAX_OMEGA = 0.5     

# pi*Gamma(q) ~ A*q^2 : fit only in [FIT_QMIN_PIGAMMA, FIT_QMAX_PIGAMMA]
FIT_QMIN_PIGAMMA = None   
FIT_QMAX_PIGAMMA = 0.5   

# ---------- Unit conversion ----------
MEV_TO_THz = 1.0 / 4.135667696   # THz per meV


# ============================================================
# Helper functions
# ============================================================
def load_dho_table(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    q = data[:, 1]
    wL, gL = data[:, 2], data[:, 3]
    wT, gT = data[:, 4], data[:, 5]
    return q, wL, gL, wT, gT


def window_mask(q, qmin=None, qmax=None):
    m = np.isfinite(q)
    if qmin is not None:
        m &= q >= qmin
    if qmax is not None:
        m &= q <= qmax
    return m


def find_crossing(q, y1, y2, qmin=None, qmax=None):
    m = window_mask(q, qmin, qmax) & np.isfinite(y1) & np.isfinite(y2)
    q2, y12, y22 = q[m], y1[m], y2[m]
    if len(q2) < 2:
        return None, None
    diff = y12 - y22
    s = np.sign(diff)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return None, None
    i = idx[0]
    q0, q1 = q2[i], q2[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    q_star = q0 - d0 * (q1 - q0) / (d1 - d0)
    y_star = np.interp(q_star, q2, y12)
    return q_star, y_star


def fit_omega_linear(q, omega, qmin, qmax):
    """
    Fit omega = v*q + b using ONLY q in [qmin, qmax].
    Returns (v, b) or (None, None).
    """
    m = window_mask(q, qmin, qmax) & np.isfinite(omega)
    qf, of = q[m], omega[m]
    if len(qf) < 2:
        return None, None
    v, b = np.polyfit(qf, of, 1)
    return v, b


def fit_pigamma_q2(q, pigamma, qmin, qmax):
    """
    Fit pigamma ≈ A*q^2 (through origin) using ONLY q in [qmin, qmax].
    Returns A or None.
    """
    m = window_mask(q, qmin, qmax) & np.isfinite(pigamma)
    qf, pgf = q[m], pigamma[m]
    if len(qf) < 2:
        return None
    x = qf**2
    denom = np.dot(x, x)
    if denom <= 0:
        return None
    A = np.dot(x, pgf) / denom
    return A


def finalize_axis(ax, q, ymax_data):
    ax.set_xlabel(r"$q$ (1/$\AA$)")
    ax.set_ylabel("Frequency (THz)")

    ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR))
    ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(which="minor", axis="y", linestyle=":", alpha=0.5)

    # y limits
    if Y_MAX is None:
        ax.set_ylim(Y_MIN, ymax_data * 1.10)
    else:
        ax.set_ylim(Y_MIN, Y_MAX)

    # x limits
    if X_MIN is None and X_MAX is None:
        ax.set_xlim(q.min() * 0.95, q.max() * 1.05)
    else:
        xmin = q.min() * 0.95 if X_MIN is None else X_MIN
        xmax = q.max() * 1.05 if X_MAX is None else X_MAX
        ax.set_xlim(xmin, xmax)

    ax.legend(frameon=False)


# ============================================================
# Main
# ============================================================
q, wL_meV, gL_meV, wT_meV, gT_meV = load_dho_table(INPUT_TXT)

# sort by q
idx = np.argsort(q)
q = q[idx]
wL_meV, gL_meV = wL_meV[idx], gL_meV[idx]
wT_meV, gT_meV = wT_meV[idx], gT_meV[idx]

# meV -> THz
wL = wL_meV * MEV_TO_THz
gL = gL_meV * MEV_TO_THz
wT = wT_meV * MEV_TO_THz
gT = gT_meV * MEV_TO_THz

pi_gL = np.pi * gL
pi_gT = np.pi * gT

# IR points
qIR_L, wIR_L = find_crossing(q, wL, pi_gL, QMIN_IR, QMAX_IR)
qIR_T, wIR_T = find_crossing(q, wT, pi_gT, QMIN_IR, QMAX_IR)

# Fits (ONLY in user-chosen low-q window)
vL, bL = fit_omega_linear(q, wL, FIT_QMIN_OMEGA, FIT_QMAX_OMEGA)
vT, bT = fit_omega_linear(q, wT, FIT_QMIN_OMEGA, FIT_QMAX_OMEGA)
AL = fit_pigamma_q2(q, pi_gL, FIT_QMIN_PIGAMMA, FIT_QMAX_PIGAMMA)
AT = fit_pigamma_q2(q, pi_gT, FIT_QMIN_PIGAMMA, FIT_QMAX_PIGAMMA)

# q-grid for plotting fit lines
qline = np.linspace(q.min(), q.max(), 400)

# ============================================================
# (1) L + T
# ============================================================
fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=160)

ax.plot(q, wL, "v", color="tab:red", label=r"$\Omega_L$")
ax.plot(q, pi_gL, "p", mfc="none", color="tab:red", label=r"$\pi\Gamma_L$")
ax.plot(q, wT, "o", color="tab:blue", label=r"$\Omega_T$")
ax.plot(q, pi_gT, "s", mfc="none", color="tab:blue", label=r"$\pi\Gamma_T$")

if vL is not None:
    ax.plot(qline, vL*qline + bL, "-", color="tab:red", alpha=0.6, label=r"fit $\Omega_L$")
if vT is not None:
    ax.plot(qline, vT*qline + bT, "-", color="tab:blue", alpha=0.6, label=r"fit $\Omega_T$")
if AL is not None:
    ax.plot(qline, AL*qline**2, ":", color="tab:red", alpha=0.7, label=r"fit $\pi\Gamma_L\propto q^2$")
if AT is not None:
    ax.plot(qline, AT*qline**2, ":", color="tab:blue", alpha=0.7, label=r"fit $\pi\Gamma_T\propto q^2$")

#if qIR_L is not None:
#    ax.axvline(qIR_L, ls="--", color="tab:red")
#    ax.axhline(wIR_L, ls="--", color="tab:red")
#if qIR_T is not None:
#    ax.axvline(qIR_T, ls="-.", color="tab:blue")
#    ax.axhline(wIR_T, ls="-.", color="tab:blue")

ymax = max(wL.max(), pi_gL.max(), wT.max(), pi_gT.max())
finalize_axis(ax, q, ymax)
fig.tight_layout()
fig.savefig(OUT_ALL, dpi=300)
plt.close(fig)

# ============================================================
# (2) L only
# ============================================================
fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=160)

ax.plot(q, wL, "v", color="tab:blue", label=r"$\Omega_L$")
ax.plot(q, pi_gL, "p", mfc="none", color="tab:red", label=r"$\pi\Gamma_L$")

if vL is not None:
    ax.plot(qline, vL*qline + bL, "-", color="tab:blue", alpha=0.6, label=r"fit $\Omega_L$")
if AL is not None:
    ax.plot(qline, AL*qline**2, ":", color="tab:red", alpha=0.7, label=r"fit $\pi\Gamma_L\propto q^2$")

if qIR_L is not None:
    ax.axvline(qIR_L, ls="--", color="tab:blue")
    ax.axhline(wIR_L, ls="--", color="tab:blue")

ymax = max(wL.max(), pi_gL.max())
finalize_axis(ax, q, ymax)
fig.tight_layout()
fig.savefig(OUT_L, dpi=300)
plt.close(fig)

# ============================================================
# (3) T only
# ============================================================
fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=160)

ax.plot(q, wT, "o", color="tab:blue", label=r"$\Omega_T$")
ax.plot(q, pi_gT, "s", mfc="none", color="tab:red", label=r"$\pi\Gamma_T$")

if vT is not None:
    ax.plot(qline, vT*qline + bT, "-", color="tab:blue", alpha=0.6, label=r"fit $\Omega_T$")
if AT is not None:
    ax.plot(qline, AT*qline**2, ":", color="tab:red", alpha=0.7, label=r"fit $\pi\Gamma_T\propto q^2$")

if qIR_T is not None:
    ax.axvline(qIR_T, ls="-.", color="tab:blue")
    ax.axhline(wIR_T, ls="-.", color="tab:blue")

ymax = max(wT.max(), pi_gT.max())
finalize_axis(ax, q, ymax)
fig.tight_layout()
fig.savefig(OUT_T, dpi=300)
plt.close(fig)

print("Saved files:")
print(" ", OUT_ALL)
print(" ", OUT_L)
print(" ", OUT_T)

if vL is not None:
    print(f"[fit] Omega_L: omega = {vL:.4f} q + {bL:.4f} (THz), q in [{FIT_QMIN_OMEGA}, {FIT_QMAX_OMEGA}]")
if vT is not None:
    print(f"[fit] Omega_T: omega = {vT:.4f} q + {bT:.4f} (THz), q in [{FIT_QMIN_OMEGA}, {FIT_QMAX_OMEGA}]")
if AL is not None:
    print(f"[fit] pi*Gamma_L: pi*Gamma = {AL:.4f} q^2 (THz), q in [{FIT_QMIN_PIGAMMA}, {FIT_QMAX_PIGAMMA}]")
if AT is not None:
    print(f"[fit] pi*Gamma_T: pi*Gamma = {AT:.4f} q^2 (THz), q in [{FIT_QMIN_PIGAMMA}, {FIT_QMAX_PIGAMMA}]")


if qIR_L is not None:
    print(f"[IR] L-mode : q_IR = {qIR_L:.4f} 1/Å , f_IR = {wIR_L:.4f} THz")
else:
    print("[IR] L-mode : crossing not found")

if qIR_T is not None:
    print(f"[IR] T-mode : q_IR = {qIR_T:.4f} 1/Å , f_IR = {wIR_T:.4f} THz")
else:
    print("[IR] T-mode : crossing not found")

