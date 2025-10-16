# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import joblib
import pickle

# (opcional pero útil en la nube para evitar exceso de threads)
torch.set_num_threads(1)


st.set_page_config(page_title="ML_MyT", layout="wide", initial_sidebar_state="expanded")
########################### bar style
# ---- Sidebar fija (ancha) ----
st.markdown("""
<style>
  /* Fuerza ancho fijo grande */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div:first-child {
      width: 720px !important;
      min-width: 720px !important;
      max-width: 720px !important;
  }

  /* Que el contenido principal no colisione visualmente */
  main .block-container {
      padding-left: 1.25rem;   /* ajusta si lo ves muy pegado */
      padding-right: 1.5rem;
  }

  /* Opcional: que todo dentro de la sidebar use el ancho disponible */
  [data-testid="stSidebarContent"] {
      padding-right: 10px;
      padding-left: 10px;
  }
</style>
""", unsafe_allow_html=True)


# ---- Estilos del hero centrado ----
st.markdown("""
<style>
  /* Contenedor general centrado */
  .block-container {
      display: flex;
      flex-direction: column;
      align-items: center;
  }

  /* Hero */
  .hero-wrap { 
      margin-top: 1rem; 
      margin-bottom: 1.5rem; 
      text-align: center; 
      width: 100%;
  }

  .hero-logo img {
      display: block;
      margin-left: auto;
      margin-right: auto;
  }

  .hero-subtitle { 
      text-align: center; 
      font-size: 0.95rem; 
      line-height: 1.55; 
      color: rgba(250,250,250,0.85);
      margin-top: .25rem;
  }

  .hero-hr { 
      border: none; 
      height: 1px; 
      background: linear-gradient(90deg, transparent, rgba(200,200,200,.25), transparent);
      margin: .75rem auto 1rem auto; 
      width: 72%;
  }

  .refs { 
      text-align: center; 
      font-size: 0.86rem; 
      color: rgba(220,220,220,0.75);
      margin-top: .25rem;
  }

  .main-block p { 
      font-size: 0.95rem; 
      line-height: 1.55; 
      text-align: justify;
      max-width: 780px;
      margin: 0 auto; 
  }
</style>
""", unsafe_allow_html=True)

# --- HERO centrado sin HTML (LaTeX funciona) ---
# Centrado con columnas
left, center, right = st.columns([0.5, 7, 0.5])
with center:
    st.image("logo.jpg",  use_container_width=True)

    st.markdown(
r"""
**ML-MYT** predicts values for four physical parameters from distant reflection X-ray spectra of Active Galactic Nuclei (AGN) observed with *NuSTAR*, using a simulation-based inference approach (*Neural Posterior Estimation*).  
The algorithm provides the **modes of the posterior distributions** and the **68% credible intervals** for the following physical parameters:

- $N_{\mathrm{H,Z}}$ [10$^{24}$ cm$^{-2}$] — (line-of-sight)  
- $N_{\mathrm{H,S}}$ [10$^{24}$ cm$^{-2}$] — (global/scattered)  
- $\Gamma$ — (photon index)  
- $A_S$ — (relative normalization)
"""
    )

    st.markdown("---")

    st.markdown(
r"""
*References* — **MYTORUS**: [Murphy & Yaqoob (2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.397.1549M/abstract); [Yaqoob (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.423.3360Y/abstract).  
*Method & training data*: Daza-Perilla *et al.* — “[AGN X-ray Reflection Spectroscopy with ML-MYTORUS: Neural Posterior Estimation with Training on Observation-Driven Parameter Grids](https://arxiv.org/list/hep-ph/recent)” (submitted).
"""
    )



########################### Load models (cached)
@st.cache_resource(show_spinner=False)
def load_scalers_and_model():
    scaler_prior = joblib.load("Model/scaler_prior.save")
    scaler_spec  = joblib.load("Model/scaler_spec.save")
    with open("Model/model.pkl", "rb") as f:
        posterior = pickle.load(f)
    return scaler_prior, scaler_spec, posterior

@st.cache_data(show_spinner=False)
def load_energy_reference(path_txt: str):
    """Reads a reference file [energy, counts] and returns indices for channels 3–30 keV."""
    ref = pd.read_csv(path_txt, names=['energy', 'counts'], header=None, sep=r"\s+")
    energies_row = pd.DataFrame(ref['energy']).T
    cols_3_30 = energies_row.loc[:, (energies_row.iloc[0] >= 3) & (energies_row.iloc[0] < 30)].columns.values
    return cols_3_30

try:
    scaler_prior, scaler_spec, posterior = load_scalers_and_model()
except Exception as e:
    st.error(f"Could not load scalers/models: {e}")
    st.stop()

# Reference file for energy bins
ENERGY_REF_PATH = "mytd-gau-ID-0.txt"

try:
    features_idx = load_energy_reference(ENERGY_REF_PATH)
except Exception as e:
    st.error(f"Could not read reference file '{ENERGY_REF_PATH}': {e}")
    st.stop()

########################### User input
with st.sidebar:
    with st.expander("How to use (inputs & expected format)"):
        st.markdown(r"""
ML-MYT estimates the physical parameters from **distant reflection X-ray spectra of Active Galactic Nuclei (AGN)m NuSTAR**
using a simulation-based inference approach.

**Steps to use:**
1. Upload a two-column ASCII file (whitespace-separated):  
   `energy_keV  counts`  
   This file should include the 4096 energy channels corresponding to the complete NuSTAR spectral range.

2. Provide the **Galactic equivalent neutral hydrogen column density** $N^{\mathrm{gal}}_{\mathrm{H}}$ [cm⁻²],  
   the redshift *z*,  
   and **total exposure time** [s].

3. The model returns the **most probable values** (posterior modes),  
   the **68% credible intervals**, and their limits  
   for the following physical parameters:  
   - $N_{\mathrm{H,Z}} [10^{24} \,\mathrm{cm}^{-2}$] — line-of-sight **equivalent neutral hydrogen column density**  
   - $N_{\mathrm{H,S}} [10^{24} \,\mathrm{cm}^{-2}$] — global (scattered) **equivalent neutral hydrogen column density**  
   - $\Gamma$ — photon index  
   - $A_S$ — relative normalization



**Important:** The input spectrum must be exported using the **XSPEC-based method** (effective energies via RMF) to ensure consistency with the model’s training distribution.
""")
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # NEW: Important (PHA → TXT tools & notes) WITH DOWNLOADS
        with st.expander("⚠️ PHA → TXT conversion tools & notes"):
            st.markdown(r"""
            **Why these scripts are required**  
            ML-MyT expects spectra exported from **XSPEC** (effective energies via the RMF),  
            while pure-Python readers (EBOUNDS-only) yield nominal energies.  
            That leads to **different numerical inputs** for the network.

            **Purpose**  
            Automatically convert a `.pha` into a two-column ASCII file:

            `energy_keV   counts`

            **How to use**
            1) Install **HEASoft/XSPEC**.  
            2) Place your `.pha`, its `.rmf` (and any referenced `.arf`/`.bkg`) in the same directory.  
            3) Run:
            This produces `yourfile.pha.asc`.

            **Post-processing inside `p2a`:**
            """)
            st.code(
                "awk '{if(NR>3) {print $1,$3}}' w1.qdp > a.asc\n"
                "mv a.asc $phafile.asc\n"
                "/bin/rm w1.qdp",
                language="bash"
                )
            st.markdown(r"""
            - `NR>3` skips the header lines in `w1.qdp`.  
            - Columns: **$1 = energy (keV)**, **$3 = counts**.  
            - Output is a clean ASCII file compatible with ML-MyT.

            **Downloads**
            Below you can download both helper scripts:
            """)

                    # Bash script (p2a)
            p2a_script = r"""#!/bin/bash
            # Panayiotis Tzanavaris 10Oct2025
            # Convert PHA → ASCII (energy, counts) using XSPEC + pha2asc.tcl

            phafile=$1
            tempfile=/tmp/${RANDOM}_run.xcm
            . $HEADAS/headas-init.sh

            echo chatter 0        > $tempfile
            echo pha2asc $phafile >> $tempfile
            echo exit             >> $tempfile

            xspec - $tempfile > /dev/null
            /bin/rm $tempfile

            awk '{if(NR>3) {print $1,$3}}' w1.qdp > a.asc
            mv a.asc $phafile.asc
            /bin/rm w1.qdp

            echo
            echo "$phafile --> $phafile.asc"
            echo
            """

                    # XSPEC Tcl script (pha2asc.tcl)
            pha2asc_tcl = r"""proc pha2asc {phafile} {
                # Panayiotis Tzanavaris 10Oct2025
                # Convert a PHA to two-column ASCII using XSPEC plotting pipeline
                file delete w1.qdp
                data $phafile
                setplot energy
                ignore **-3.0 30.-**
                setplot command wdata w1
                plot counts
            }"""

            st.download_button("⬇️ Download `p2a` (bash)", data=p2a_script, file_name="p2a", mime="text/plain")
            st.download_button("⬇️ Download `pha2asc.tcl` (XSPEC Tcl)", data=pha2asc_tcl, file_name="pha2asc.tcl", mime="text/plain")
            st.caption("Requires HEASoft/XSPEC available in $HEADAS (RMF must be accessible via RESPFILE).")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    st.header("Input parameters")
    nhzgal = st.number_input("N_H,Gal", min_value=0.0, step=0.1, value=0.0)
    z = st.number_input("Redshift (z)", min_value=0.0, step=0.001, value=0.0, format="%.3f")
    exposure_time = st.number_input("Total exposure time (s)", min_value=0.0, step=1.0, value=0.0)

    

    uploaded_file = st.file_uploader("Upload your spectrum (2 columns: energy, counts)", type=["txt", "dat", "csv"])
    if uploaded_file is None:
        st.info("Waiting for file...")
        st.stop()

########################### Read spectrum
try:
    df = pd.read_csv(uploaded_file, names=['energy', 'counts'], header=None, sep=r"\s+")
except Exception as e:
    st.error(f"Could not read uploaded file: {e}")
    st.stop()

if not set(['energy','counts']).issubset(df.columns):
    st.error("The file must contain two columns: 'energy' and 'counts' (no header).")
    st.stop()

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Preview of file")
    st.dataframe(df.head(20), use_container_width=True)


import altair as alt

with col2:
    st.subheader("Uploaded spectrum")
    st.caption(
        "Interactive visualization of the uploaded NuSTAR spectrum. "
        "The shaded orange region highlights the 3–30 keV energy range "
        "used for ML-MyT training and inference."
    )

    chart = (
        alt.Chart(df)
        .mark_line(color="#64B5F6")  # azul suave
        .encode(
            x=alt.X("energy", title="Energy (keV)"),
            y=alt.Y("counts", title="Counts"),
            tooltip=["energy", "counts"]
        )
        .properties(width="container", height=350)
        .interactive()
    )

    band = alt.Chart(pd.DataFrame({'start':[3], 'end':[30]})).mark_rect(
    opacity=0.1, color='orange').encode(x='start:Q', x2='end:Q')
    st.altair_chart(band + chart, use_container_width=True)



########################### Preprocessing
df_T = pd.DataFrame(df['counts']).T

if df_T.shape[1] <= np.max(features_idx):
    st.error(
        "Uploaded spectrum has fewer channels than required by the 3–30 keV mask.\n"
        f"Shape received: {df_T.shape[1]} — max required index: {np.max(features_idx)}"
    )
    st.stop()

# Select features (3–30 keV channels)
spec = df_T[features_idx].copy()

# Add auxiliary features (make sure these match your training setup)
spec['DOUBLE_EXP_TIME_(s)'] = exposure_time
spec['z'] = z
spec['nH_G'] = nhzgal

feature_order = list(spec.columns)

# Scale to float32
spec = spec.astype(np.float32)
try:
    spec_scaled = scaler_spec.transform(spec.values)
except Exception as e:
    st.error(
        "Failed to transform with scaler_spec. Check that columns and order match training.\n"
        f"Current columns: {feature_order}\nError: {e}"
    )
    st.stop()

# Observation vector (D,)
x_obs = torch.from_numpy(spec_scaled.squeeze(0))

########################### Posterior sampling
st.subheader("Physical parameter inference")
st.markdown(
    """
Below are the **posterior mode estimates** (approximate) and **68% credible intervals** for the four 
physical parameters inferred from the uploaded spectrum.  
All values correspond to the decoupled **MYTORUS** model.

| Parameter | Description | Units |
|------------|--------------|--------|
| N_H_Z | Line-of-sight column density | 10²⁴ cm⁻² |
| N_H_S | Global (scattered) column density | 10²⁴ cm⁻² |
| Γ | Photon index | dimensionless |
| A_S | Relative normalization | dimensionless |
"""
)

@st.cache_data(show_spinner=False)
def posterior_samples(_posterior, x_obs_np, num_samples=5000):
    """Sample from posterior. `_posterior` is ignored for hashing to avoid Streamlit cache errors."""
    x_obs = torch.from_numpy(x_obs_np)
    with torch.no_grad():
        samples = _posterior.sample((num_samples,), x=x_obs)
    return samples.cpu().numpy()

# Sampling
x_obs_np = x_obs.detach().cpu().numpy()
samples_np = posterior_samples(posterior, x_obs_np, num_samples=5000)
samples_original = scaler_prior.inverse_transform(samples_np)

param_names = ['N_H_Z', 'N_H_S', 'Gamma', 'A_S']

# Mode and 68% credible interval
modes, lower_68, upper_68, int_68 = [], [], [], []
for j in range(samples_original.shape[1]):
    param_samples = samples_original[:, j]
    hist, bin_edges = np.histogram(param_samples, bins=50)
    mode_idx = np.argmax(hist)
    mode_val = 0.5 * (bin_edges[mode_idx] + bin_edges[mode_idx + 1])
    modes.append(mode_val)
    lo = np.percentile(param_samples, 16)
    hi = np.percentile(param_samples, 84)
    lower_68.append(lo)
    upper_68.append(hi)
    int_68.append(hi - lo)

results_df = pd.DataFrame({
    "Parameter": param_names,
    "Mode (posterior)": [f"{v:.4f}" for v in modes],
    "68% credible interval [p16–p84]": [f"[{l:.4f}, {u:.4f}]" for l, u in zip(lower_68, upper_68)],
    "Width (p84–p16)": [f"{w:.4f}" for w in int_68]
})

st.dataframe(results_df, use_container_width=True)

# ---------- Posterior plots (one panel per parameter) ----------
import matplotlib.pyplot as plt
from math import ceil

# Etiquetas con unidades
pretty_label = {
    'N_H_Z': r'$N_{H,Z}$ (10$^{24}$ cm$^{-2}$)',
    'N_H_S': r'$N_{H,S}$ (10$^{24}$ cm$^{-2}$)',
    'Gamma': r'$\Gamma$',
    'A_S': r'$A_S$',
}

n_params = samples_original.shape[1]
ncols = 2
nrows = ceil(n_params / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), constrained_layout=True)
axes = np.atleast_1d(axes).ravel()

for j in range(n_params):
    ax = axes[j]
    s = samples_original[:, j]

    # Histograma (densidad)
    ax.hist(s, bins=50, density=True, color='royalblue', alpha=0.85)

    # Banda 68% CI
    ax.axvspan(lower_68[j], upper_68[j], alpha=0.25, label="68% CI", color='royalblue')

    # Moda
    ax.axvline(modes[j], linewidth=2, label="Mode", color='white')

    # Etiquetas
    ax.set_xlabel(pretty_label.get(param_names[j], param_names[j]))
    ax.set_ylabel("Posterior")

    # Guía visual
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

# Ocultar ejes sobrantes si los hay
for k in range(n_params, len(axes)):
    fig.delaxes(axes[k])

st.pyplot(fig)


########################### Download results
csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results (CSV)",
    data=csv_bytes,
    file_name="physical_parameter.csv",
    mime="text/csv",
)

st.caption("Note: make sure the input columns order matches the one used during training.")
