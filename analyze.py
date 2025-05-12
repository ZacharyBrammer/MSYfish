import io
import os
import time

import netCDF4 as nc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotting import plot_simulation


# Method to allow plotting/viewing of previous simulations
def analyze(labels):
    # Select which simulation to plot
    folders = [""] + sorted(os.listdir("simulations/"))
    folder = st.selectbox(label="Folder", options=folders)
    
    # Select folder
    if folder != "":
        # Get all species in the folder. If there's multiple species, allow choice
        speciess = [""] + os.listdir(f"simulations/{folder}")
        if len(speciess) > 2:
            species = st.selectbox(label="Species", options=sorted(os.listdir(f"simulations/{folder}")))
        else:
            species = speciess[-1]
        
        # Get all simulations in the folder and let user select
        simulations = [""] + os.listdir(f"simulations/{folder}/{species}")
        simulation = st.selectbox(label="Simulation", options=simulations)
    else:
        # If blank folder is selected set species and simulation to blank
        species = simulation = ""

    if simulation != "":
        # Read the dataset
        path = f"simulations/{folder}/{species}/{simulation}"
        biodata = nc.Dataset(path, "r")

        # Setup for reading individual variables in file
        # TODO: allow user to make table from whatever variales they want, turn that into plots
        for variable in biodata.variables:
            print(variable, ":",biodata.variables[variable].__dict__["long_name"])
        st.json(biodata.variables, expanded=1)
        st.selectbox(label="Variables", options=biodata.variables)

        # Plot the simulation
        plots = plot_simulation(path)
        for plot in plots:
            st.plotly_chart(plot)
        
        
