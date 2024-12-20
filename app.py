import io
import json
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

# Language setup
languages = ["en", "es"]
with open("labels.json", "r") as f:
    labels = json.load(f)
    f.close()

# Get species data from spreadsheet, can turn into user uploaded file later
fishdata = pd.read_excel("fish_growth_data2.xlsx")
speciesList = fishdata["species"]

st.title("MSYFish Model")

# Set up session state
if "running" not in st.session_state:
    st.session_state.running = False

if "language" not in st.session_state:
    st.session_state.language = "en"

with st.sidebar:
    language = st.selectbox(label="Language", options=languages, disabled=st.session_state.running)

    # If language changed, rerun page
    if st.session_state.language != language:
        st.session_state.language = language
        st.rerun()

# Run button
runButton = st.button(label=labels["run"][st.session_state.language])

if runButton:
    st.session_state.running = True

# Have user select species, get indexes for running model
selectedSpecies: list[str] = st.multiselect(label=labels["species"][st.session_state.language], options=speciesList, disabled=st.session_state.running)
speciesIndexes: list[int] = []

for species in selectedSpecies:
    speciesIndexes.append(speciesList[speciesList == species].index.values[0].item())

# Get output directory
directory = st.text_input(label=labels["output_dir"][st.session_state.language], value="path/", disabled=st.session_state.running)

# Integer inputs to the model
stocks = st.number_input(label=labels["stocks"][st.session_state.language], step=1, min_value=1, disabled=st.session_state.running)
niter = st.number_input(label=labels["iterations"][st.session_state.language], step=1, min_value=1, disabled=st.session_state.running)
years = st.number_input(label=labels["years"][st.session_state.language], step=1, min_value=1, disabled=st.session_state.running)
initialPop = st.number_input(label=labels["initial_pop"][st.session_state.language], step=1, min_value=1, disabled=st.session_state.running)


# Optional enables
st.write(labels["optional_param"][st.session_state.language])

# Load connectivity file
conn_file = st.file_uploader(label=labels["conn_file"][st.session_state.language], type=["xls", "xlsx"], disabled=st.session_state.running)
if conn_file is not None:
    file_bytes = io.BytesIO(conn_file.getvalue())
    conn_data = pd.read_excel(file_bytes)
    connectivity = conn_data.to_numpy()
    connectivity = connectivity[:,1:]

    # Check that connectivity file matches number of stocks
    if not (connectivity.shape[0] == stocks and connectivity.shape[1] == stocks):
        # Warn user, set connectivity to none
        st.warning(labels["conn_warning"][st.session_state.language])
        connectivity = np.array(None)
        conn_file = None
else:
    connectivity = np.array(None)

fishing = st.toggle(label=labels["fishing"][st.session_state.language], disabled=st.session_state.running)
rotation = st.toggle(label=labels["rotation"][st.session_state.language], disabled=st.session_state.running)

# Check that stocks > 1 to prevent divide by 0 error
if rotation and stocks == 1:
    st.warning(labels["rotation_warning"][st.session_state.language])
    rotation = False

# Run model
if st.session_state.running:
    for i in range(len(speciesIndexes)):
        calc_msy(directory, fishdata, connectivity, i, stocks, niter, years, initialPop, fishing, rotation)

        # If final run, re-enable inputs
        if i == len(speciesIndexes) - 1:
            st.session_state.running = False
            st.success(labels["sim_complete"][st.session_state.language])
            time.sleep(5)
            st.rerun()
