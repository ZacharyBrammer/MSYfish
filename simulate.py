import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from calc_msy_rotational_serve import calc_msy
from plotting import plot_simulation


# Method to both display simulation settings and run simulations
def simulate(labels):
    # Get species data from spreadsheet, can turn into user uploaded file later
    fishdata = pd.read_excel("fish_growth_data2.xlsx")
    speciesList = fishdata[st.session_state.names]

    # Run button
    runButton = st.button(label=labels["run"][st.session_state.language])

    if runButton:
        st.session_state.running = True

    # Have user select species, get indexes for running model
    # TODO: add switching for scientific/common names
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
    # If fishing is enabled, let user set fishing rate as a percentage of calculated max value
    if fishing:
        fishingRate = st.number_input(label=labels["fishing_rate"][st.session_state.language], min_value=0.0, max_value=100.0, value=100.0, disabled=st.session_state.running)
        
    else:
        fishingRate = 0

    sizes = st.toggle(label=labels["sizes"][st.session_state.language], disabled=st.session_state.running)
    # If size select is enabled, let user set a minimum and maximum catch size
    if sizes:
        minCatchSize = st.number_input(label=labels["min_catch_size"][st.session_state.language], min_value=0.0, value=0.0, disabled=st.session_state.running)
        maxCatchSize = st.number_input(label=labels["max_catch_size"][st.session_state.language], min_value=minCatchSize, value=None, disabled=st.session_state.running)
    else:
        minCatchSize = 0
        maxCatchSize = 0

    rotation = st.toggle(label=labels["rotation"][st.session_state.language], disabled=st.session_state.running)
    # If rotation is enabled, let user set rotation rate
    if rotation:
        rotationRate = st.number_input(label=labels["rotation_rate"][st.session_state.language], min_value=1, max_value=years + 100, disabled=st.session_state.running)
    else:
        rotationRate = 0

    tempEnable = st.toggle(label=labels["temp_enable"][st.session_state.language], disabled=st.session_state.running)
    # If temperature impact on production is enabled, let user set in degrees C
    if tempEnable:
        temperature = st.number_input(label=labels["temperature"][st.session_state.language], value=20.0, disabled=st.session_state.running)
    else:
        temperature = None

    # Check that stocks > 1 to prevent divide by 0 error
    if rotation and stocks == 1:
        st.warning(labels["rotation_warning"][st.session_state.language])
        rotation = False
    # If rotation rate is the same as number of years, add warning about permanent closure
    if rotationRate == (years + 100):
        st.warning(labels["rotation_rate_warning"][st.session_state.language])
    # If fishing is disabled but catch size is set
    if not fishing and sizes:
        st.warning((labels["size_warning"][st.session_state.language]))
        sizes = False

    # Run model
    if st.session_state.running:
        # Make it so not selecting a species and hitting run doesn't break everything
        if (len(speciesIndexes)) == 0:
            st.session_state.running = False
            st.rerun()
        
        for i in range(len(speciesIndexes)):
            # 100 is added to the number of years so the simulation is given time to stabilize
            calc_msy(directory, fishdata, connectivity, speciesIndexes[i], stocks, niter, (years + 100), initialPop, fishing, fishingRate, rotation, rotationRate, sizes, minCatchSize, maxCatchSize, temperature)

            # If final run, re-enable inputs and plot first run
            if i == len(speciesIndexes) - 1:
                # Get path to first simulation to plot
                firstSimDir = os.getcwd() + "/simulations/" + directory + "/" + os.listdir("simulations/" + directory)[0]
                print(directory, firstSimDir)
                firstSimPath = firstSimDir + "/" + os.listdir(firstSimDir)[0]
                st.session_state.plot = plot_simulation(firstSimPath)

                # Set running to false and print success message
                st.session_state.running = False
                st.success(labels["sim_complete"][st.session_state.language])
                time.sleep(5)
                st.rerun()

    # Display images and other data from sim
    if st.session_state.plot != "":
        st.write(st.session_state.fishingDat)
        st.write(st.session_state.popDat)
        for plot in st.session_state.plot:
            st.plotly_chart(plot)
