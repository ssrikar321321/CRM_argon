import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import streamlit as st

# Predefined NIST + Ar 2p transition data with fixed upper states
nist_data = pd.DataFrame({
    "Wavelength_nm": [
        965.78, 912.3, 1047.0, 1148.8,
        811.53, 801.48, 842.47, 978.45,
        772.38, 810.37, 866.79, 935.42,
        763.51, 800.62, 922.45,
        751.47,
        852.14, 794.82, 747.12, 714.70,
        706.72, 738.40, 840.82,
        696.54, 727.29, 772.42, 826.45,
        750.39, 667.73
    ],
    "Aki_s-1": [
        5.4e6, 1.9e7, 1.0e6, 1.9e5,
        3.3e7, 9.3e6, 2.2e7, 1.5e6,
        5.0e6, 2.5e7, 2.4e6, 1.0e6,
        2.5e7, 4.9e6, 5.0e6,
        4.0e7,
        1.4e7, 1.9e7, 2.2e4, 6.25e5,
        3.8e6, 8.7e6, 2.2e7,
        6.4e6, 1.8e6, 1.2e7, 1.5e7,
        4.5e7, 2.36e5
    ],
    "Upper_State": [
        "2p10", "2p10", "2p10", "2p10",
        "2p9", "2p8", "2p8", "2p8",
        "2p7", "2p7", "2p7", "2p7",
        "2p6", "2p6", "2p6",
        "2p5", "2p4", "2p4", "2p4", "2p4",
        "2p3", "2p3", "2p3",
        "2p2", "2p2", "2p2", "2p2",
        "2p1", "2p1"
    ],
    "E_upper_eV": [
        14.4, 14.3, 14.6, 14.1,
        14.0, 13.6, 14.1, 14.0,
        13.5, 13.7, 13.3, 13.6,
        13.4, 13.3, 13.7,
        13.5, 13.2, 13.1, 13.0, 13.0,
        13.3, 13.5, 13.4,
        13.1, 13.2, 13.2, 13.3,
        13.0, 13.1
    ]
})

st.title("Argon Spectral Analysis with Fixed Upper States and Updated NIST Data")

# File uploaders
measurement_file = st.file_uploader("Measurement Spectrum (.asc, .txt)", type=["asc", "txt"])
background_file = st.file_uploader("Background Spectrum (optional)", type=["asc", "txt"])
pop_file = st.file_uploader("Ar 2p Population Data (.txt)", type=["txt"])

# Input parameters
peak_offset = st.number_input("Peak offset window (±nm)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
thresh = st.number_input("Peak detection threshold (normalized)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

# Additional inputs for line ratio plot
i1 = st.number_input("Intensity 1", value=1.0)
i2 = st.number_input("Intensity 2", value=1.0)

if measurement_file:
    # Load measurement
    meas = pd.read_csv(measurement_file, delim_whitespace=True, header=None,skiprows=list(range(0,20)))
    wl_meas, inten_meas = meas[0].values, meas[1].values

    # Subtract background
    if background_file:
        bg = pd.read_csv(background_file, delim_whitespace=True, header=None,skiprows=list(range(0,20)))
        inten_meas = np.clip(inten_meas - bg[1].values, 0, None)

    # Peak detection
    norm_int = inten_meas / inten_meas.max()
    peaks, _ = find_peaks(norm_int, height=thresh)
    wl_peaks, inten_peaks = wl_meas[peaks], inten_meas[peaks]

    # Plot measured spectrum
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(wl_meas, inten_meas, label="Measurement")
    ax.plot(wl_peaks, inten_peaks, 'rx', label="Detected Peaks")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(650,810)
    ax.set_title("Measured Spectrum with Detected Peaks")
    ax.legend()
    st.pyplot(fig)

    # Match peaks to NIST data
    matched = []
    for wl, inten in zip(wl_peaks, inten_peaks):
        diff = np.abs(nist_data['Wavelength_nm'] - wl)
        idx = diff.idxmin()
        if diff[idx] <= peak_offset:
            row = nist_data.loc[idx]
            matched.append({
                'Measured Peak (nm)': wl,
                'Intensity': inten,
                'NIST Wavelength (nm)': row['Wavelength_nm'],
                'Aki (s^-1)': row['Aki_s-1'],
                'Upper State': row['Upper_State'],
                'E_upper (eV)': row['E_upper_eV']
            })

    if matched:
        df_matched = pd.DataFrame(matched)
        st.subheader("Matched Peaks with Fixed Upper States")
        st.dataframe(df_matched)

        # Select peaks for final analysis
        sel = st.multiselect("Select peaks for analysis", options=df_matched.index,
                             format_func=lambda i: f"{df_matched.loc[i, 'NIST Wavelength (nm)']:.2f} nm")

        if sel:
            final = df_matched.loc[sel].reset_index(drop=True)
            final['Normalized Intensity'] = final['Intensity'] / final['Intensity'].sum()
            final['Upper_Index'] = final['Upper State'].str.extract(r'2p(\d+)').astype(int)

            st.subheader("Final Selection for CR Analysis")
            st.dataframe(final)

            # Theoretical intensity comparison
            if pop_file:
                pop = np.loadtxt(pop_file)
                Te_min = st.number_input("Electron Temperature Range: Start (eV)", min_value=0.01, value=0.5)
                Te_max = st.number_input("Electron Temperature Range: End (eV)", min_value=Te_min+0.01, value=2.0)
                Te_vals = np.linspace(Te_min, Te_max, pop.shape[1])

                A = final['Aki (s^-1)'].values
                idxs = final['Upper_Index'].values - 1  # zero-based

                I_th = (pop[idxs, :].T * A).T  # shape (n_lines, n_Te)
                I_th_norm = I_th / I_th.sum(axis=0)
                I_exp = final['Normalized Intensity'].values[:, None]

                # Compute RMS deviation
                rms = np.sqrt(((I_th_norm - I_exp)**2).mean(axis=0))
                best = rms.argmin()

                st.subheader(f"Best Match Electron Temperature: {Te_vals[best]:.3f} eV (RMS: {rms[best]:.2e})")

                # Plot results
                fig2, axes = plt.subplots(1,2, figsize=(14,5))
                axes[0].plot(Te_vals, rms, '-o')
                axes[0].set_yscale('log')
                axes[0].set_xlabel('Electron Temperature (eV)')
                axes[0].set_ylabel('RMS Deviation')
                axes[0].set_title('RMS vs Temperature')

                x = np.arange(len(final))
                axes[1].bar(x-0.2, I_exp.flatten(), 0.4, label='Experiment')
                axes[1].bar(x+0.2, I_th_norm[:, best], 0.4, label='CR Model')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(final['NIST Wavelength (nm)'].round(1), rotation=45)
                axes[1].set_yscale('log')
                axes[1].set_xlabel('Wavelength (nm)')
                axes[1].set_ylabel('Normalized Intensity')
                axes[1].set_title('Experimental vs Theoretical')
                axes[1].legend()

                fig2.tight_layout()
                st.pyplot(fig2)
                
                # Plot results
                fig3, axes = plt.subplots(1,2, figsize=(14,5))
                axes[0].plot(Te_vals, rms, '-o')
                #axes[0].set_yscale('log')
                axes[0].set_xlabel('Electron Temperature (eV)')
                axes[0].set_ylabel('RMS Deviation')
                axes[0].set_title('RMS vs Temperature')

                x = np.arange(len(final))
                axes[1].bar(x-0.2, I_exp.flatten(), 0.4, label='Experiment')
                axes[1].bar(x+0.2, I_th_norm[:, best], 0.4, label='CR Model')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(final['NIST Wavelength (nm)'].round(1), rotation=45)
                #axes[1].set_yscale('log')
                axes[1].set_xlabel('Wavelength (nm)')
                axes[1].set_ylabel('Normalized Intensity')
                axes[1].set_title('Experimental vs Theoretical')
                axes[1].legend()
                fig3.tight_layout()
                st.pyplot(fig3)

                # Output comparison table
                comp = pd.DataFrame({
                    'Wavelength (nm)': final['NIST Wavelength (nm)'],
                    'Experimental': I_exp.flatten(),
                    'Theoretical': I_th_norm[:, best]
                })
                st.write(comp)
                # --- New Boltzmann Plot ---
                # Boltzmann plot: ln(Intensity/Aki) vs E_upper
                energies = final['E_upper (eV)'].values
                lnI = np.log(final['Intensity'].values / final['Aki (s^-1)'].values)
                # Linear fit
                slope, intercept = np.polyfit(energies, lnI, 1)
                # Estimate temperature (K) from slope: slope = -1/(k_B * T)
                k_B = 8.617e-5  # eV/K
                T_est = -1 / (slope * k_B)

                fig4, ax4 = plt.subplots(figsize=(7,5))
                ax4.scatter(energies, lnI, label='Data')
                ax4.plot(energies, slope*energies + intercept, '--', label=f'Fit (T~{T_est:.0f} K)')
                ax4.set_xlabel('E_upper (eV)')
                ax4.set_ylabel('ln(Intensity / Aki)')
                ax4.set_title('Boltzmann Plot for Electron Temperature')
                ax4.legend()
                st.pyplot(fig3)

                # --- Line Ratio Plot ---
                ratio = i1 / i2 if i2 != 0 else np.nan
                fig5, ax5 = plt.subplots(figsize=(7,5))
                ax5.bar(['I1', 'I2'], [i1, i2])
                ax5.set_ylabel('Intensity')
                ax5.set_title(f'Line Intensities and Ratio I1/I2 = {ratio:.2f}')
                st.pyplot(fig4)
    else:
        st.warning("No peaks matched within the offset window.")
