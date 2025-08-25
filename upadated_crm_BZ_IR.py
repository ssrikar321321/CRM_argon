#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argon Spectral Analysis – Ordered Diagnostics
============================================
User‑requested order:
1. **CR‑model comparison (raw intensities)**
2. **Boltzmann plot (raw)**
3. **SACF‑corrected Boltzmann plot**
4. **CR‑model comparison (SACF‑corrected intensities)**

Electron‑temperature range input (Te_min / Te_max) retained from original code.
Updated on 2025‑08‑25 by ChatGPT.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

K_B = 8.617e-5  # Boltzmann constant (eV/K)

# ------------------------------------------------------------------
# NIST reference (unchanged)
# ------------------------------------------------------------------
nist_data = pd.DataFrame({
    "Wavelength_nm": [965.78, 912.3, 1047.0, 1148.8, 811.53, 801.48, 842.47, 978.45,
                      772.38, 810.37, 866.79, 935.42, 763.51, 800.62, 922.45, 751.47,
                      852.14, 794.82, 747.12, 714.70, 706.72, 738.40, 840.82, 696.54,
                      727.29, 772.42, 826.45, 750.39, 667.73],
    "Aki_s-1": [5.4e6, 1.9e7, 1.0e6, 1.9e5, 3.3e7, 9.3e6, 2.2e7, 1.5e6,
                5.0e6, 2.5e7, 2.4e6, 1.0e6, 2.5e7, 4.9e6, 5.0e6, 4.0e7,
                1.4e7, 1.9e7, 2.2e4, 6.25e5, 3.8e6, 8.7e6, 2.2e7, 6.4e6,
                1.8e6, 1.2e7, 1.5e7, 4.5e7, 2.36e5],
    "Upper_State": ["2p10", "2p10", "2p10", "2p10", "2p9", "2p8", "2p8", "2p8",
                   "2p7", "2p7", "2p7", "2p7", "2p6", "2p6", "2p6", "2p5",
                   "2p4", "2p4", "2p4", "2p4", "2p3", "2p3", "2p3", "2p2",
                   "2p2", "2p2", "2p2", "2p1", "2p1"],
    "E_upper_eV": [14.4, 14.3, 14.6, 14.1, 14.0, 13.6, 14.1, 14.0, 13.5, 13.7,
                   13.3, 13.6, 13.4, 13.3, 13.7, 13.5, 13.2, 13.1, 13.0, 13.0,
                   13.3, 13.5, 13.4, 13.1, 13.2, 13.2, 13.3, 13.0, 13.1]
})

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def SACF(I, I1, A, A1, G, G1, E, E1, etemp):
    fb = (I / I1) * (A1 / A) * (G1 / G) * np.exp((E - E1) / etemp)
    return fb if fb < 1 else 1


def download_dataframe(df: pd.DataFrame, label: str):
    csv = df.to_csv(index=False).encode()
    st.download_button(f"Download {label} (CSV)", csv, file_name=f"{label}.csv", mime="text/csv")

# ------------------------------------------------------------------
# Streamlit layout
# ------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Argon Spectral Analysis (Ordered Diagnostics)")

# --- Uploaders & parameters ---
measurement_file = st.file_uploader("Measurement Spectrum (.asc | .txt)", ["asc", "txt"])
background_file = st.file_uploader("Background Spectrum (optional)", ["asc", "txt"])
pop_file = st.file_uploader("Ar 2p Population Matrix (.txt)", ["txt"])

colA, colB = st.columns(2)
with colA:
    thresh = st.slider("Peak detection threshold", 0.0, 1.0, 0.2)
with colB:
    peak_offset = st.slider("Peak‑match tolerance (nm)", 0.01, 1.0, 0.3)

with st.sidebar:
    st.header("Electron‑Temperature Range (CR model)")
    Te_min = st.number_input("Te start (eV)", min_value=0.01, value=0.5)
    Te_max = st.number_input("Te end (eV)", min_value=Te_min + 0.01, value=2.0)
    st.markdown("---")
    st.header("Plot axis limits (optional)")
    x_Te_min = st.number_input("Te axis min", value=Te_min)
    x_Te_max = st.number_input("Te axis max", value=Te_max)
    y_rms_min = st.number_input("RMS axis min", value=1e-4)
    y_rms_max = st.number_input("RMS axis max", value=1.0)

# ------------------------------------------------------------------
# Measurement processing
# ------------------------------------------------------------------
if measurement_file:
    meas = pd.read_csv(measurement_file, delim_whitespace=True, header=None, skiprows=range(20))
    wl_meas, inten_meas = meas[0].values, meas[1].values

    if background_file:
        bg = pd.read_csv(background_file, delim_whitespace=True, header=None, skiprows=range(20))
        inten_meas = np.clip(inten_meas - bg[1].values, 0, None)

    norm_int = inten_meas / inten_meas.max()
    peaks, _ = find_peaks(norm_int, height=thresh)
    wl_peaks, inten_peaks_raw = wl_meas[peaks], inten_meas[peaks]

    inten_peaks_norm = inten_peaks_raw / inten_peaks_raw.sum()

    # match peaks
    matched = []
    for wl, inten in zip(wl_peaks, inten_peaks_norm):
        diff = np.abs(nist_data["Wavelength_nm"] - wl)
        idx = diff.idxmin()
        if diff[idx] <= peak_offset:
            row = nist_data.loc[idx]
            matched.append({
                "Measured Peak (nm)": wl,
                "Intensity": inten,
                "NIST Wavelength (nm)": row["Wavelength_nm"],
                "Aki (s^-1)": row["Aki_s-1"],
                "Upper State": row["Upper_State"],
                "E_upper (eV)": row["E_upper_eV"]
            })

    if not matched:
        st.error("No peaks matched. Adjust threshold/tolerance.")
        st.stop()

    df_match = pd.DataFrame(matched)
    st.subheader("Matched Peaks")
    st.dataframe(df_match)

    sel = st.multiselect("Select peaks for analysis", options=df_match.index,
                         format_func=lambda i: f"{df_match.loc[i,'NIST Wavelength (nm)']:.2f} nm")
    if not sel:
        st.info("Please select peaks.")
        st.stop()

    final = df_match.loc[sel].reset_index(drop=True)
    st.subheader("Final Selection")
    st.dataframe(final)

    # ---------------------------------------------------------------------------------
    # === 1) CR model comparison (RAW intensities) ===
    # ---------------------------------------------------------------------------------
    if pop_file:
        st.markdown("## 1. CR Model (raw intensities)")
        pop = np.loadtxt(pop_file)
        Te_vals = np.linspace(Te_min, Te_max, pop.shape[1])

        upper_idx = final["Upper State"].str.extract(r"2p(\d+)").astype(int)[0].values - 1
        A_vals = final["Aki (s^-1)"].values
        I_exp_raw = final["Intensity"].values   # already normalised

        I_theo = (pop[upper_idx, :].T * A_vals).T
        I_theo_norm = I_theo / I_theo.sum(axis=0)

        rms_raw = np.sqrt(((I_theo_norm - I_exp_raw[:, None])**2).mean(axis=0))
        best_raw = int(np.argmin(rms_raw))
        Te_best_raw = Te_vals[best_raw]

        st.info(f"Best match Te (RAW) = {Te_best_raw:.3f} eV  |  RMS = {rms_raw[best_raw]:.3e}")

        fig_rms_raw, ax_rms_raw = plt.subplots()
        ax_rms_raw.plot(Te_vals, rms_raw, "-o")
        ax_rms_raw.set_xlabel("Te (eV)")
        ax_rms_raw.set_ylabel("RMS deviation")
        ax_rms_raw.set_yscale("log")
        ax_rms_raw.set_xlim(x_Te_min, x_Te_max)
        ax_rms_raw.set_ylim(y_rms_min, y_rms_max)
        ax_rms_raw.set_title("CR model RMS vs Te (raw)")
        st.pyplot(fig_rms_raw)

        # intensity comparison at best Te (raw)
        fig_cmp_raw, ax_cmp_raw = plt.subplots()
        x = np.arange(len(final))
        ax_cmp_raw.bar(x-0.2, I_exp_raw, 0.4, label="Experimental")
        ax_cmp_raw.bar(x+0.2, I_theo_norm[:, best_raw], 0.4, label="CR model")
        ax_cmp_raw.set_xticks(x)
        ax_cmp_raw.set_xticklabels(final["NIST Wavelength (nm)"].round(1), rotation=45)
        ax_cmp_raw.set_yscale("log")
        ax_cmp_raw.set_ylabel("Normalised intensity")
        ax_cmp_raw.set_title("CR vs Exp (RAW)")
        ax_cmp_raw.legend()
        st.pyplot(fig_cmp_raw)

        download_dataframe(pd.DataFrame({"Te (eV)": Te_vals, "RMS": rms_raw}), "RMS_raw")

    # ---------------------------------------------------------------------------------
    # === 2) Boltzmann plot (RAW) ===
    # ---------------------------------------------------------------------------------
    st.markdown("## 2. Boltzmann Plot (raw)")
    Aki = final["Aki (s^-1)"].values
    E = final["E_upper (eV)"].values
    g = np.full_like(Aki, 3)
    y_raw_bz = np.log(final["Intensity"].values / (g * Aki))
    lr_raw_bz = LinearRegression().fit(E.reshape(-1,1), y_raw_bz)
    Te_raw_bz = -1 / lr_raw_bz.coef_[0]

    fig_bz_raw, ax_bz_raw = plt.subplots()
    ax_bz_raw.scatter(E, y_raw_bz, label="Data")
    ax_bz_raw.plot(E, lr_raw_bz.predict(E.reshape(-1,1)), "--", label=f"Fit Te={Te_raw_bz:.2f} eV")
    ax_bz_raw.set_xlabel("E_upper (eV)")
    ax_bz_raw.set_ylabel("ln(I/gA)")
    ax_bz_raw.legend()
    st.pyplot(fig_bz_raw)

    # ---------------------------------------------------------------------------------
    # === 3) SACF‑corrected Boltzmann ===
    # ---------------------------------------------------------------------------------
    st.markdown("## 3. SACF‑Corrected Boltzmann Plot")
    ref_index = st.selectbox("Reference line for SACF", options=final.index,
                             format_func=lambda i: f"{final.loc[i,'NIST Wavelength (nm)']:.2f} nm")

    I_raw_vec = final["Intensity"].values
    I_ref = I_raw_vec[ref_index]
    E_ref = E[ref_index]
    A_ref = Aki[ref_index]
    G_ref = g[ref_index]

    z_fac = np.array([SACF(I_raw_vec[i], I_ref, Aki[i], A_ref, g[i], G_ref, E[i], E_ref, Te_raw_bz)
                      for i in range(len(I_raw_vec))])
    I_corr = I_raw_vec / z_fac

    y_corr_bz = np.log(I_corr / (g * Aki))
    lr_corr_bz = LinearRegression().fit(E.reshape(-1,1), y_corr_bz)
    Te_corr_bz = -1 / lr_corr_bz.coef_[0]

    fig_bz_corr, ax_bz_corr = plt.subplots()
    ax_bz_corr.scatter(E, y_corr_bz, color="blue", label="SACF data")
    ax_bz_corr.plot(E, lr_corr_bz.predict(E.reshape(-1,1)), "b-", label=f"Fit Te={Te_corr_bz:.2f} eV")
    ax_bz_corr.set_xlabel("E_upper (eV)")
    ax_bz_corr.set_ylabel("ln(I/gA)")
    ax_bz_corr.legend()
    st.pyplot(fig_bz_corr)

    st.success(f"Te (Boltzmann raw) = {Te_raw_bz:.2f} eV | Te (SACF) = {Te_corr_bz:.2f} eV")

    # ---------------------------------------------------------------------------------
    # === 4) CR model comparison (SACF‑corrected intensities) ===
    # ---------------------------------------------------------------------------------
    if pop_file:
        st.markdown("## 4. CR Model (SACF‑corrected intensities)")
        I_corr_norm = I_corr / I_corr.sum()
        I_theo_corr = (pop[upper_idx, :].T * A_vals).T
        I_theo_corr_norm = I_theo_corr / I_theo_corr.sum(axis=0)

        rms_corr = np.sqrt(((I_theo_corr_norm - I_corr_norm[:, None])**2).mean(axis=0))
        best_corr = int(np.argmin(rms_corr))
        Te_best_corr = Te_vals[best_corr]

        st.info(f"Best match Te (SACF) = {Te_best_corr:.3f} eV  |  RMS = {rms_corr[best_corr]:.3e}")

        fig_rms_corr, ax_rms_corr = plt.subplots()
        ax_rms_corr.plot(Te_vals, rms_corr, "-o")
        ax_rms_corr.set_xlabel("Te (eV)")
        ax_rms_corr.set_ylabel("RMS deviation")
        ax_rms_corr.set_yscale("log")
        ax_rms_corr.set_xlim(x_Te_min, x_Te_max)
        ax_rms_corr.set_ylim(y_rms_min, y_rms_max)
        ax_rms_corr.set_title("CR model RMS vs Te (SACF)")
        st.pyplot(fig_rms_corr)

        # comparison plot
        fig_cmp_corr, ax_cmp_corr = plt.subplots()
        x = np.arange(len(final))
        ax_cmp_corr.bar(x-0.2, I_corr_norm, 0.4, label="SACF‑corrected")
        ax_cmp_corr.bar(x+0.2, I_theo_corr_norm[:, best_corr], 0.4, label="CR model")
        ax_cmp_corr.set_xticks(x)
        ax_cmp_corr.set_xticklabels(final["NIST Wavelength (nm)"].round(1), rotation=45)
        ax_cmp_corr.set_yscale("log")
        ax_cmp_corr.set_ylabel("Normalised intensity")
        ax_cmp_corr.set_title("CR vs SACF‑Corrected")
        ax_cmp_corr.legend()
        st.pyplot(fig_cmp_corr)

        download_dataframe(pd.DataFrame({"Te (eV)": Te_vals, "RMS": rms_corr}), "RMS_SACF")
