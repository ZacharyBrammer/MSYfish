import io
import os
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

from calc_msy_rotational_serve import calc_msy
from plotting import plot_simulation
from translate import Translator


# Method to both display simulation settings and run simulations
def simulate():
    translator = Translator(st.session_state.language)
    t = translator.translate

    # Get species data from spreadsheet, can turn into user uploaded file later
    fishdata = pd.read_excel("fish_growth_data2.xlsx")
    speciesList = fishdata[st.session_state.names]

    # Have user select species, get index for running model
    selectedSpecies = st.selectbox(
        label=t("species"),
        options=speciesList,
        disabled=st.session_state.running
    )
    speciesIndex = speciesList[speciesList == selectedSpecies].index[0]

    dir_regex = r"^[A-Za-z0-9_-]+$"

    # Get output directory
    directory = st.text_input(
        label=t("output_dir"),
        value="path",
        disabled=st.session_state.running
    )

    if not re.fullmatch(dir_regex, directory):
        st.session_state.valid_path = False
        st.error(
            "Invalid folder name. Only letters, numbers, underscores, and hyphens are allowed.")
    else:
        st.session_state.valid_path = True

    initButton = st.button("Initialize Simulation")
