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
    if st.session_state.initd:
        st.write(f"Maximum Calculated Fishing Rate: {st.session_state.sim.maxfish:.2f}")

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

        # connectivity
        conn_enable = st.toggle(
            label="Connectivity",
            disabled=st.session_state.running
        )
        if conn_enable:
            if "num_stocks" not in st.session_state:
                st.session_state.num_stocks = stocks

            if "conn_matrix" not in st.session_state or stocks != st.session_state.num_stocks:
                st.session_state.conn_matrix = pd.DataFrame(
                    np.zeros((stocks, stocks), dtype=float),
                    columns=[f"Stock {i+1}" for i in range(stocks)],
                    index=[f"Stock {i+1}" for i in range(stocks)]
                )
            # TODO: add editor like for temperature
            conn_file = st.file_uploader(
                label=t("conn_file"),
                type=["xls", "xlsx", "csv"],
                disabled=st.session_state.running
            )
            if conn_file:
                # TODO: add more verification, make sure values are in proper range (ask Dr Woodson)
                # Read file contents
                if conn_file.name.endswith(".csv"):
                    df = pd.read_csv(conn_file, index_col=0)
                else:  # xls or xlsx
                    df = pd.read_excel(conn_file, index_col=0)

                # check that connectivity file matches number of stocks
                if df.shape != (stocks, stocks):
                    # warn user, set connectivity to none
                    st.warning(t("conn_warning"))
                    connectivity = np.array(None)
                    conn_file = None
                else:
                    # Validate that rows sum to 1
                    rowSums = df.sum(axis=1).round(6)
                    if not np.allclose(rowSums, 1.0, atol=1e-6):
                        st.warning("Rows must sum up to 1")
                    else:
                        st.session_state.conn_matrix = df
            
            # Editable matrix for connectivity
            st.session_state.conn_matrix = st.data_editor(
                st.session_state.conn_matrix,
                disabled=st.session_state.running
            )

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
            if "years" not in st.session_state:
                st.session_state.years = years
            
            # set up temperature data to be number of years with default value of 20C
            if "temps" not in st.session_state or years != st.session_state.years:
                st.session_state.temps = np.hstack(
                [np.arange(1, years + 1).reshape(-1, 1),
                 np.full((years, 1), 20.0)]
            )
            
            # temperature file upload
            temp_file = st.file_uploader(
                label="Temperature File",
                type=["xls", "xlsx", "csv"],
                disabled=st.session_state.running
            )

            if temp_file:
                # read file contents
                if temp_file.name.endswith(".csv"):
                    df = pd.read_csv(temp_file, index_col=0)
                else:  # xls or xlsx
                    df = pd.read_excel(temp_file, index_col=0)
                
                # check that tempearature file matches years
                if df.shape != (years, 1):
                    # warn user, set connectivity to none
                    st.warning(t("conn_warning"))
                    temperature = np.array(None)
                    temp_file = None
                else:
                    st.session_state.temps = df

            # data editor to allow users to input their temperature data
            st.session_state.temps = st.data_editor(
                st.session_state.temps,
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

            temperature = st.session_state.temps.values.ravel()
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
            # connectivity validation
            if conn_enable:
                # convert conn_matrix to an ndarray, do final validation
                connectivity = st.session_state.conn_matrix.to_numpy()
                # validate that rows sum to 1
                rowSums = connectivity.sum(axis=1).round(6)
                if np.allclose(connectivity, 0):
                    connectivity = np.array(None)
                elif not np.allclose(rowSums, 1.0, atol=1e-6):
                    st.warning("Rows must sum up to 1, disabling connectivity")
                    connectivity = np.array(None)
            else:
                connectivity = np.array(None)

            
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
