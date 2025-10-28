# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import joblib
import pickle
import altair as alt
from scipy.stats import gaussian_kde
from math import ceil

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
      padding-left: 1.25rem;
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
  .block-container {
      display: flex;
      flex-direction: column;
      align-items: center;
  }
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

# --- HERO centrado ---
left, center, right = st.columns([0.5, 7, 0.5])
with center:
    st.image("logo.jpg",  use_container_width=True)
    st.markdown(
r"""
**ML-MYT** predicts values for four physical parameters from distant reflection X-ray spectra of Active Galactic Nuclei (AGN) observed with *NuSTAR*, using a simulation-based inference approach (*Neural Posterior Estimation*).  
The algorithm provides the **posterior modes** (predictive point estimates) and the **credible intervals** for:

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
*Method & training data*: Daza-Perilla *et al.* — “AGN X-ray Reflection Spectroscopy with ML-MYTORUS: Neural Posterior Estimation with Training on Observation-Driven Parameter Grids” (submitted).
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
ML-MYT estimates physical parameters from **NuSTAR** distant reflection X-ray spectra (AGN) using **Neural Posterior Estimation (SBI)**.

**Steps:**
1. Upload a whitespace-separated two-column ASCII file:  
   `energy_keV  counts` with **4096 channels** (full NuSTAR range).
2. Provide $N^{\mathrm{gal}}_{\mathrm{H}}$ [cm⁻²], the redshift *z*, and total exposure time [s].
3. The model returns the **posterior mode** and the **68% / 90% credible intervals**.
""")
        with st.expander("⚠️ PHA → TXT conversion tools & notes"):
            st.markdown(r"""
ML-MyT expects spectra exported from **XSPEC** using effective energies via the **RMF**, ensuring physical consistency with the training data.  
Pure-Python readers that rely only on **EBOUNDS** produce nominal (uncorrected) energies, which do not match the calibrated inputs used for training.

**Purpose:** convert `.pha` → ASCII: `energy_keV   counts`
""")
            st.code(
                "awk '{if(NR>3) {print $1,$3}}' w1.qdp > a.asc\n"
                "mv a.asc $phafile.asc\n"
                "/bin/rm w1.qdp",
                language="bash"
            )
            p2a_script = r"""#!/bin/bash
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
            pha2asc_tcl = r"""proc pha2asc {phafile} {
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

    st.header("Input parameters")
    nhzgal = st.number_input("N_H,Gal", min_value=0.0, step=0.1, value=0.0)
    z = st.number_input("Redshift (z)", min_value=0.0, step=0.001, value=0.0, format="%.5f")
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

with col2:
    st.subheader("Uploaded spectrum")
    st.caption(
        "Interactive visualization of the uploaded NuSTAR spectrum. "
        "The shaded orange region highlights the 3–30 keV energy range used for ML-MyT."
    )
    chart = (
        alt.Chart(df)
        .mark_line(color="#64B5F6")
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

########################### Posterior sampling (Bayesian)
st.subheader("Physical parameter inference (Bayesian)")
st.markdown(
    """
We report the **posterior modes** (predictive point estimates) and **credible intervals**:
- **68%**: central [p16–p84]  
- **90%**: central [p5–p95]  

All values correspond to the decoupled **MYTORUS** model.
"""
)

@st.cache_data(show_spinner=False)
def posterior_samples(_posterior, x_obs_np, num_samples=5000):
    """Sample the learned posterior p(theta|x_obs)."""
    x_obs_t = torch.from_numpy(x_obs_np)
    with torch.no_grad():
        samples = _posterior.sample((num_samples,), x=x_obs_t)
    return samples.cpu().numpy()

# --- Sampling ---
x_obs_np = x_obs.detach().cpu().numpy()
samples_np = posterior_samples(posterior, x_obs_np, num_samples=5000)
samples_original = scaler_prior.inverse_transform(samples_np)

# ------- Utilidades bayesianas -------
def kde_mode_1d(samples: np.ndarray) -> float:
    """Posterior mode via KDE (robust wrt binning)."""
    kde = gaussian_kde(samples)
    grid = np.linspace(np.min(samples), np.max(samples), 4096)
    pdf = kde(grid)
    return float(grid[np.argmax(pdf)])

def credible_interval_percentiles(samples: np.ndarray, level: float) -> tuple[float,float]:
    """Central credible interval by percentiles."""
    lo_q = (1 - level) / 2 * 100.0
    hi_q = (1 + level) / 2 * 100.0
    lo, hi = np.percentile(samples, [lo_q, hi_q])
    return float(lo), float(hi)

param_names = ['N_H_Z', 'N_H_S', 'Gamma', 'A_S']
modes, lo68, hi68, lo90, hi90 = [], [], [], [], []

for j in range(len(param_names)):
    s = samples_original[:, j]
    modes.append(kde_mode_1d(s))
    l68, h68 = credible_interval_percentiles(s, 0.68)  # p16–p84
    l90, h90 = credible_interval_percentiles(s, 0.90)  # p5–p95
    lo68.append(l68); hi68.append(h68)
    lo90.append(l90); hi90.append(h90)

# Tabla (solo bayesiano)
results_df = pd.DataFrame({
    "Parameter": param_names,
    "Posterior mode (predictive)": [f"{v:.6g}" for v in modes],
    "68% credible interval [p16–p84]": [f"[{a:.6g}, {b:.6g}]" for a,b in zip(lo68, hi68)],
    "90% credible interval [p5–p95]":  [f"[{a:.6g}, {b:.6g}]" for a,b in zip(lo90, hi90)],
})
st.dataframe(results_df, use_container_width=True)

# ---------- Posterior plots ----------
st.subheader("Posterior distributions per parameter")
pretty_label = {
    'N_H_Z': r'$N_{H,Z}$ (10$^{24}$ cm$^{-2}$)',
    'N_H_S': r'$N_{H,S}$ (10$^{24}$ cm$^{-2}$)',
    'Gamma': r'$\Gamma$',
    'A_S': r'$A_S$',
}

n_params = len(param_names)
ncols = 2
nrows = ceil(n_params / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), constrained_layout=True)
axes = np.atleast_1d(axes).ravel()

for j, name in enumerate(param_names):
    ax = axes[j]
    s = samples_original[:, j]

    # Histograma como densidad (sin estilos de color específicos)
    ax.hist(s, bins=50, density=True)

    # Bandas: 90% más ancha, 68% interna
    ax.axvspan(lo90[j], hi90[j], alpha=0.15, label="90% credible")
    ax.axvspan(lo68[j], hi68[j], alpha=0.30, label="68% credible")

    # Moda (estimador puntual)
    ax.axvline(modes[j], linewidth=2, label="Posterior mode")

    ax.set_xlabel(pretty_label.get(name, name))
    ax.set_ylabel("Density")
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
    file_name="physical_parameters_bayes.csv",
    mime="text/csv",
)

# Nota metodológica breve (sin mezclar con XSPEC aquí)
st.caption(
    "Bayesian output: posterior mode (point estimate) and 68%/90% credible intervals. "
    "Credible intervals reflect probability over parameters given the observed data. "
    "Frequentist XSPEC intervals (e.g., 90% CL via ΔC=2.706) are confidence intervals "
    "with long-run coverage. Matching 68/90 levels enables fair visual comparison, "
    "but interpretations remain different (Bayesian vs Frequentist)."
)
