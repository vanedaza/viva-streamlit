# app.py
import pandas as pd
import numpy as np
import streamlit as st
import torch
import joblib
import pickle

# (opcional pero útil en la nube para evitar exceso de threads)
torch.set_num_threads(1)

st.set_page_config(page_title="ViVa Mi Pana", layout="wide")

########################### Title
st.title("ViVa Mi Pana :sunglasses:")
st.text(
    "To use VIVA, upload a CSV/TXT file with two whitespace-separated columns:\n"
    "   1) energy (keV)   2) counts\n"
)

########################### Load models (cached)
@st.cache_resource(show_spinner=False)
def load_scalers_and_model():
    scaler_prior = joblib.load("Model/scaler_prior_app6_gau.save")
    scaler_spec  = joblib.load("Model/scaler_spec_app6_gau.save")
    with open("Model/model_app6_gau.pkl", "rb") as f:
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
    st.header("Input parameters")
    nhzgal = st.number_input("N_H,Gal", min_value=0.0, step=0.1, value=0.0)
    z = st.number_input("Redshift (z)", min_value=0.0, step=0.001, value=0.0, format="%.3f")
    exposure_time = st.number_input("Double exposure time (s)", min_value=0.0, step=1.0, value=0.0)

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
    st.line_chart(df, x="energy", y="counts")

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

@st.cache_data(show_spinner=False)
def posterior_samples(_posterior, x_obs_np, num_samples=5000):
    """Sample from posterior. `_posterior` is ignored for hashing to avoid Streamlit cache errors."""
    x_obs = torch.from_numpy(x_obs_np)  # rebuild CPU tensor inside
    with torch.no_grad():
        samples = _posterior.sample((num_samples,), x=x_obs)
    return samples.cpu().numpy()

# Pass numpy to the cache (important)
x_obs_np = x_obs.detach().cpu().numpy()

try:
    samples_np = posterior_samples(posterior, x_obs_np, num_samples=5000)
except Exception as e:
    st.error(f"Posterior sampling failed: {e}")
    st.stop()

# Inverse transform to original parameter space
try:
    samples_original = scaler_prior.inverse_transform(samples_np)
except Exception as e:
    st.error(f"Failed to inverse-transform with scaler_prior: {e}")
    st.stop()

param_names = ['N_H_Z', 'N_H_S', 'Gamma', 'A_S']
if samples_original.shape[1] != len(param_names):
    param_names = [f"param_{i}" for i in range(samples_original.shape[1])]

# Mode and 68% interval
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
    "parameter": param_names,
    "mode_approx": modes,
    "p16": lower_68,
    "p84": upper_68,
    "int68": int_68,
})

st.dataframe(results_df, use_container_width=True)

########################### Download results
csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results (CSV)",
    data=csv_bytes,
    file_name="physical_parameter.csv",
    mime="text/csv",
)

st.caption("Note: make sure the input columns order matches the one used during training.")
