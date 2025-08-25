import io
import os
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

from plotting import plot_simulation
from simulator import Simulator
from translate import Translator


# method to both display simulation settings and run simulations
def simulate():
    translator = Translator(st.session_state.language)
    t = translator.translate

    # get species data from spreadsheet, can turn into user uploaded file later
    fishdata = pd.read_excel("fish_growth_data2.xlsx")
    speciesList = fishdata[st.session_state.names]

    # have user select species, get index for running model
    selectedSpecies = st.selectbox(
        label=t("species"),
        options=speciesList,
        disabled=st.session_state.running
    )
    speciesIndex = speciesList[speciesList == selectedSpecies].index[0]

    dir_regex = r"^[A-Za-z0-9_-]+$"

    # get output directory
    directory = st.text_input(
        label=t("output_dir"),
        value="path",
        disabled=st.session_state.running
    )

    # do regex to make sure path is valid
    if not re.fullmatch(dir_regex, directory):
        st.session_state.valid_path = False
        st.error(
            "Invalid folder name. Only letters, numbers, underscores, and hyphens are allowed.")
    else:
        st.session_state.valid_path = True

    initButton = st.button(
        "Initialize Simulation",
        disabled=not st.session_state.valid_path or st.session_state.initd
    )
    if initButton:
        st.session_state.init = True

    # initialize the model for the selected species
    if st.session_state.init and not st.session_state.initd:
        # TODO: add a check to see if simulator object exists for species already
        st.session_state.sim = Simulator(
            outdir=directory,
            fishdata=fishdata,
            speciesIndex=speciesIndex
        )
        st.session_state.initd = True

    # TODO: display max fishing rate here

    # get inputs to the model for running
    if st.session_state.initd:
        # integer inputs to the model
        stocks = st.number_input(
            label=t("stocks"),
            step=1,
            min_value=1,
            disabled=st.session_state.running
        )
        numiter = st.number_input(
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

        # optional enables
        st.write(t("optional_param"))

        # load connectivity file
        # TODO: add editor like for temperature
        conn_file = st.file_uploader(
            label=t("conn_file"),
            type=["xls", "xlsx"],
            disabled=st.session_state.running
        )
        if conn_file is not None:
            # TODO: add more verification, make sure values are in proper range (ask Dr Woodson)
            file_bytes = io.BytesIO(conn_file.getvalue())
            conn_data = pd.read_excel(file_bytes)
            connectivity = conn_data.to_numpy()
            connectivity = connectivity[:, 1:]

            # check that connectivity file matches number of stocks
            if not (connectivity.shape[0] == stocks and connectivity.shape[1] == stocks):
                # warn user, set connectivity to none
                st.warning(t("conn_warning"))
                connectivity = np.array(None)
                conn_file = None
        else:
            connectivity = np.array(None)

        # fishing stuff
        fishing = st.toggle(
            label=t("fishing"),
            disabled=st.session_state.running
        )

        # if fishing is enabled, show relevant settings
        if fishing:
            fishingRate = st.number_input(
                label=t("fishing_rate"),
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                disabled=st.session_state.running
            )

            sizes = st.toggle(
                label=t("sizes"),
                disabled=st.session_state.running
            )
            # if size select is enabled, let user set a minimum and maximum catch size
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
            # if rotation is enabled, let user set rotation rate
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
            sizes = False
            minCatchSize = 0
            maxCatchSize = 0
            rotation = False
            rotationRate = 0

        # temperature stuff
        tempEnable = st.toggle(
            label=t("temp_enable"),
            disabled=st.session_state.running
        )

        # TODO: add file upload stuff like in connectivity matrix
        # if temperature impact on production is enabled, let user set in degrees C
        if tempEnable:
            # set up temperature data to be number of years with default value of 20C
            temps = np.hstack(
                [np.arange(1, years + 1).reshape(-1, 1),
                 np.full((years, 1), 20.0)]
            )

            # data editor to allow users to input their temperature data
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

        # climatic events
        climaticEnable = st.toggle(
            label="Climatic Event Enable",
            disabled=st.session_state.running
        )

        if climaticEnable:
            massChance = st.number_input(
                label="% Chance of Climatic Event",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                disabled=st.session_state.running
            )
            massMort = st.number_input(
                label="% Population Lost In Event",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                disabled=st.session_state.running
            )
        else:
            massChance = None
            massMort = None

        # warnings
        # TODO: look for any other places user could mess up input, add errors for
        # check that stocks > 1 to prevent divide by 0 error
        if rotation and stocks == 1:
            st.warning(t("rotation_warning"))
            rotation = False
        # if rotation rate is the same as number of years, add warning about permanent closure
        if rotationRate == (years + 100):
            st.warning(t("rotation_rate_warning"))

        # run button
        runButton = st.button(
            label=t("run"), disabled=not st.session_state.valid_path)
        if runButton:
            st.session_state.running = True

        if st.session_state.running:
            # updates the directory
            st.session_state.sim.change_outdir(directory)

            for i in range(numiter):
                st.session_state.sim.simulate(connectivity=connectivity, stocks=stocks, years=years + 100, fishingRate=fishingRate, rotationRate=rotationRate,
                                              sizes=sizes, minCatch=minCatchSize, maxCatch=maxCatchSize, temperature=temperature, massChance=massChance, massMort=massMort)

                # If final run, re-enable inputs and plot first run
                if i == numiter - 1:
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
            st.write(st.session_state.popDat)
            for plot in st.session_state.plot:
                st.plotly_chart(plot)
