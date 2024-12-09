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
runButton = st.button(label="Ejecutar Simulacion")

if runButton:
    st.session_state.running = True

# Have user select species, get indexes for running model
selectedSpecies: list[str] = st.multiselect(label="Especies", options=speciesList, disabled=st.session_state.running)
speciesIndexes: list[int] = []

for species in selectedSpecies:
    speciesIndexes.append(speciesList[speciesList == species].index.values[0].item())

# Get output directory
directory = st.text_input(label="Carpeta de salida", value="path/", disabled=st.session_state.running)

# Integer inputs to the model
stocks = st.number_input(label="Poblaciones", step=1, min_value=1, disabled=st.session_state.running)
niter = st.number_input(label="Iteraciones", step=1, min_value=1, disabled=st.session_state.running)
years = st.number_input(label="Anos por simulacion", step=1, min_value=1, disabled=st.session_state.running)
initialPop = st.number_input(label="Poblacion inicial", step=1, min_value=1, disabled=st.session_state.running)


# Optional enables
st.write("Opciones")
#enableConn = st.toggle(label="Connectivity", disabled=st.session_state.running) - take num stocks x num stocks matrix
enableConn = False # connectivity doesn't actually work, run into an out of bounds error on line 332 in connect file
fishing = st.toggle(label="Pesca", disabled=st.session_state.running)
#rotation = st.toggle(label="Rotation", disabled=st.session_state.running) # ask about enabling rotation - line 163 in rotational serve - set rotation array to 0

# Load connectivity file
if enableConn:
    conn_data = pd.read_excel("connectivity.xlsx")
    connectivity = conn_data.to_numpy()
    connectivity = connectivity[:,1:]
else:
    connectivity = np.array(None)

# Run model
if st.session_state.running:
    for i in range(len(speciesIndexes)):
        calc_msy(directory, fishdata, connectivity, i, stocks, niter, years, initialPop, fishing, rotation)

        # If final run, re-enable inputs
        if i == len(speciesIndexes) - 1:
            st.session_state.running = False
            st.success("Simulations complete")
            time.sleep(5)
            st.rerun()
