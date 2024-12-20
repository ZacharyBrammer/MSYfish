import io
import time

import numpy as np
import pandas as pd
import streamlit as st

from calc_msy_rotational_serve import calc_msy

# Page setup
st.set_page_config(
    page_title="MSYFish",
    page_icon="🎣"
)

# Get species data from spreadsheet, can turn into user uploaded file later
fishdata = pd.read_excel("fish_growth_data2.xlsx")
speciesList = fishdata["species"]

st.title("MSYFish Model")

# Set up session state
if "running" not in st.session_state:
    st.session_state.running = False

# Run button
runButton = st.button(label="Run Simluations")

if runButton:
    st.session_state.running = True

# Have user select species, get indexes for running model
selectedSpecies: list[str] = st.multiselect(label="Species", options=speciesList, disabled=st.session_state.running)
speciesIndexes: list[int] = []

for species in selectedSpecies:
    speciesIndexes.append(speciesList[speciesList == species].index.values[0].item())

# Get output directory
directory = st.text_input(label="Model Output Directory", value="path/", disabled=st.session_state.running)

# Integer inputs to the model
stocks = st.number_input(label="Number of stocks", step=1, min_value=1, disabled=st.session_state.running)
niter = st.number_input(label="Iterations to run", step=1, min_value=1, disabled=st.session_state.running)
years = st.number_input(label="Years per simulation", step=1, min_value=1, disabled=st.session_state.running)
initialPop = st.number_input(label="Initial population", step=1, min_value=1, disabled=st.session_state.running)


# Optional enables
st.write("Optional Parameters")

# Load connectivity file
conn_file = st.file_uploader(label="Connectivity Matrix", type=['xls', 'xlsx'], disabled=st.session_state.running)
if conn_file is not None:
    file_bytes = io.BytesIO(conn_file.getvalue())
    conn_data = pd.read_excel(file_bytes)
    connectivity = conn_data.to_numpy()
    connectivity = connectivity[:,1:]

    # Check that connectivity file matches number of stocks
    if not (connectivity.shape[0] == stocks and connectivity.shape[1] == stocks):
        # Warn user, set connectivity to none
        st.warning("Connectivity file does not match number of stocks, connectivity set to none. Update number of stocks or choose new connectivity file.")
        connectivity = np.array(None)
        conn_file = None
else:
    connectivity = np.array(None)

fishing = st.toggle(label="Fishing", disabled=st.session_state.running)
rotation = st.toggle(label="Rotation", disabled=st.session_state.running)

# Run model
if st.session_state.running:
    for i in range(len(speciesIndexes)):
        #calc_msy(directory, fishdata, connectivity, i, stocks, niter, years, initialPop, fishing, rotation)

        # If final run, re-enable inputs
        if i == len(speciesIndexes) - 1:
            st.session_state.running = False
            st.success("Simulations complete")
            time.sleep(5)
            st.rerun()
