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

        # Get years. If simulation ended early, get index of last year
        years = biodata.variables["ftime"][:].data[100:]
        endsEarly = np.where(years == 0)[0].size != 0
        if endsEarly:
            last = np.where(years == 0)[0][0]
        else:
            last = None

        # Subtract 100 from all years to remove stabilization period
        years -= 100
        years = years[:last]

        # Download button for the simulation file
        with open(path, "rb") as file:
            st.download_button(
                label="Download NetCDF file",
                data=file,
                file_name=f"{simulation}",
                icon=":material/download:",
            )

        # Setup for reading individual variables in file
        # TODO: allow user to make table from whatever variales they want, turn that into plots
        variable_desc = {}
        for variable in biodata.variables:
            variable_desc[variable] = biodata.variables[variable].__dict__["long_name"]
        
        
        # TODO: Put this all in a toggle
        st.write("Create custom table/plot:")
        st.write("Dataset Variables")
        st.json(variable_desc, expanded=0)

        # Remove variables that aren't time-related/would be difficult for users to make sense of
        selectable_vars = biodata.variables.copy()
        selectable_vars.pop("ftime")
        selectable_vars.pop("fishing_rate")
        selectable_vars.pop("reprod_rate")
        selectable_vars.pop("rotation_rate")
        selectable_vars.pop("age")
        selectable_vars.pop("mid_bins")
        selectable_vars.pop("fish") # Too many things for a user to reasonably be able to see on a plot

        variables = st.multiselect(label="Select Variables", options=selectable_vars)

        # TODO: for each selected: get title, get data, if more than 1d do a for loop to add all bins/stock with label
        # Collect data user selected, start with time variable
        selected_data = pd.DataFrame()
        selected_data["year"] = years
        for variable in variables:
            # If 1D, add data. If 2D, title each bin and add. If 3D, add the bins for each stock then add next stock
            match len(biodata.variables[variable].shape):
                case 1:
                    # Get the data for the variable, along with variable name and units
                    data = biodata.variables[variable][:].data[100:][:last]
                    title = f"{variable} ({biodata.variables[variable].getncattr("units")})"

                    # Add data to the dataframe
                    selected_data[title] = data
                case 2:
                    # Get the data for the variable
                    data = biodata.variables[variable][:].data[100:][:last]

                    # Some 2D variables have different units so handle differently
                    match variable:
                        case "stock_size" | "stock_biomass" | "stock_recruit":
                            data = pd.DataFrame(data, columns=[f"stock {i + 1} {variable.split("_")[1]} (# of individuals)" for i in range(data.shape[1])])
                            selected_data = pd.concat([selected_data, data], axis=1)
                        case "reproduction" | "mortality":
                            data = pd.DataFrame(data, columns=[f"{variable} biomass (kg)", f"{variable} number (#)"])
                            selected_data = pd.concat([selected_data, data], axis=1)
                        case "resource":
                            print(data.shape)
                            data = pd.DataFrame(data, columns=[f"stock {i + 1} resource (units)" for i in range(data.shape[1])])
                            selected_data = pd.concat([selected_data, data], axis=1)
                case 3:
                    pass
                case _:
                    pass

        selected_data = selected_data.set_index("year")
        st.dataframe(selected_data)


        # Plot the simulation (regular plots)
        plots = plot_simulation(path)
        for plot in plots:
            st.plotly_chart(plot)
        
        
