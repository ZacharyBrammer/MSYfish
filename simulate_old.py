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

    # Run button
    runButton = st.button(label=t("run"))

    if runButton:
        st.session_state.running = True

    # Have user select species, get indexes for running model
    selectedSpecies = st.multiselect(
        label=t("species"),
        options=speciesList,
        disabled=st.session_state.running
    )
    speciesIndexes = []

    for species in selectedSpecies:
        speciesIndexes.append(
            speciesList[speciesList == species].index.values[0].item())

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

    # Integer inputs to the model
    stocks = st.number_input(
        label=t("stocks"),
        step=1,
        min_value=1,
        disabled=st.session_state.running
    )
    niter = st.number_input(
        label=t("iterations"),
        step=1,
        min_value=1,
        disabled=st.session_state.running
    )
    years = st.number_input(
        label=t("years"),
        step=1,
        min_value=1,
        disabled=st.session_state.running
    )

    # Optional enables
    st.write(t("optional_param"))

    # Load connectivity file
    conn_file = st.file_uploader(
        label=t("conn_file"),
        type=["xls", "xlsx"],
        disabled=st.session_state.running
    )
    if conn_file is not None:
        file_bytes = io.BytesIO(conn_file.getvalue())
        conn_data = pd.read_excel(file_bytes)
        connectivity = conn_data.to_numpy()
        connectivity = connectivity[:, 1:]

        # Check that connectivity file matches number of stocks
        if not (connectivity.shape[0] == stocks and connectivity.shape[1] == stocks):
            # Warn user, set connectivity to none
            st.warning(t("conn_warning"))
            connectivity = np.array(None)
            conn_file = None
    else:
        connectivity = np.array(None)

    fishing = st.toggle(
        label=t("fishing"),
        disabled=st.session_state.running
    )
    # If fishing is enabled, let user set fishing rate as a percentage of calculated max value. Other fishing settings are also hidden under this enable
    if fishing:
        biomassFishing = st.toggle(
            label="Fish by percentage of biomass",
            disabled=st.session_state.running
        )
        if biomassFishing:
            fishingRate = st.number_input(
                label="Percentage of biomass to fish",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                disabled=st.session_state.running
            )
        else:
            fishingRate = st.number_input(
                label=t("fishing_rate"),
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                disabled=st.session_state.running
            )

        sizes = st.toggle(
            label=t("sizes"),
            disabled=st.session_state.running
        )

        # If size select is enabled, let user set a minimum and maximum catch size
        if sizes:
            minCatchSize = st.number_input(
                label=t("min_catch_size"),
                min_value=0.0,
                value=0.0,
                disabled=st.session_state.running
            )
            maxCatchSize = st.number_input(
                label=t("max_catch_size"),
                min_value=minCatchSize,
                value=None,
                disabled=st.session_state.running
            )
        else:
            minCatchSize = 0
            maxCatchSize = 0

        rotation = st.toggle(
            label=t("rotation"),
            disabled=st.session_state.running
        )

        # If rotation is enabled, let user set rotation rate
        if rotation:
            rotationRate = st.number_input(
                label=t("rotation_rate"),
                min_value=1,
                max_value=years + 100,
                disabled=st.session_state.running
            )
        else:
            rotationRate = 0
    else:
        fishingRate = 0
        biomassFishing = False
        sizes = False
        minCatchSize = 0
        maxCatchSize = 0
        rotation = False
        rotationRate = 0

    tempEnable = st.toggle(
        label=t("temp_enable"),
        disabled=st.session_state.running
    )
    # If temperature impact on production is enabled, let user set in degrees C
    if tempEnable:
        # Set up temperature data to be number of years with default value of 20C
        temps = np.hstack(
            [np.arange(1, years + 1).reshape(-1, 1), np.full((years, 1), 20.0)]
        )

        # Data editor to allow users to input their temperature data
        temps = st.data_editor(
            temps,
            column_config={
                "0": st.column_config.NumberColumn(
                    "Year",
                    disabled=True
                ),
                "1": st.column_config.NumberColumn(
                    "Temperature (C)",
                    format="%.2f"
                )
            },
            disabled=st.session_state.running
        )

        temperature = temps[:, 1]
    else:
        temperature = np.array(None)

    climaticEnable = st.toggle(
        label="Climatic Event Enable",
        disabled=st.session_state.running
    )

    if climaticEnable:
        eventChance = st.number_input(
            label="% Chance of Climatic Event",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            disabled=st.session_state.running
        )
        eventMort = st.number_input(
            label="% Population Lost In Event",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            disabled=st.session_state.running
        )
    else:
        eventChance = None
        eventMort = None

    # Check that stocks > 1 to prevent divide by 0 error
    if rotation and stocks == 1:
        st.warning(t("rotation_warning"))
        rotation = False
    # If rotation rate is the same as number of years, add warning about permanent closure
    if rotationRate == (years + 100):
        st.warning(t("rotation_rate_warning"))
    # If fishing is disabled but catch size is set
    if not fishing and sizes:
        st.warning(t("size_warning"))
        sizes = False

    # Run model
    if st.session_state.running:
        # Make it so not selecting a species and hitting run doesn't break everything
        if (len(speciesIndexes)) == 0:
            st.session_state.running = False
            st.rerun()

        if st.session_state.valid_path == False:
            st.session_state.running = False
            st.rerun()

        for i in range(len(speciesIndexes)):
            # 100 is added to the number of years so the simulation is given time to stabilize
            calc_msy(directory, fishdata, connectivity, speciesIndexes[i], stocks, niter, (
                years + 100), fishing, fishingRate, rotation, rotationRate, sizes, minCatchSize, maxCatchSize, temperature, eventChance, eventMort, biomassFishing)

            # If final run, re-enable inputs and plot first run
            if i == len(speciesIndexes) - 1:
                # Get path to most recent simulation to plot
                path = f"simulations/{st.session_state.id}/{directory}"
                allSims = [
                    os.path.join(path, species, file)
                    for species in os.listdir(path)
                    for file in os.listdir(os.path.join(path, species))
                ]
                st.session_state.firstSimPath = max(
                    allSims, key=os.path.getmtime)
                st.session_state.plot = plot_simulation(
                    st.session_state.firstSimPath)

                # Set running to false and print success message
                st.session_state.running = False
                st.success(t("sim_complete"))
                time.sleep(5)
                st.rerun()

    # Display images and other data from sim
    if st.session_state.plot != "":
        simPathStr = "/".join(st.session_state.firstSimPath.split("/")[2:])
        st.write(f"{t("simulation")}: {simPathStr}")
        st.write(st.session_state.fishingDat)
        st.write(st.session_state.popDat)
        for plot in st.session_state.plot:
            st.plotly_chart(plot)
