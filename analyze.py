import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import streamlit as st

from plotting import plot_simulation
from translate import Translator


# Method to allow plotting/viewing of previous simulations
def analyze():
    translator = Translator(st.session_state.language)
    t = translator.translate
    v = translator.var

    # Select which simulation to plot
    base = f"simulations/{st.session_state.id}/"
    folders = [""] + sorted(os.listdir(base))
    folder = st.selectbox(label=t("folder"), options=folders)

    # Select folder
    if folder != "":
        # Get all species in the folder. If there's multiple species, allow choice
        speciess = [""] + os.listdir(f"{base}/{folder}")
        species = st.selectbox(label=t("species"), options=speciess)

        if species != "":
            # Get all simulations in the folder and let user select
            simulations = [""] + os.listdir(f"{base}/{folder}/{species}")
            simulation = st.selectbox(label=t("simulation"), options=simulations)
        else:
            simulation = ""
    else:
        # If blank folder is selected set species and simulation to blank
        species = simulation = ""

    if simulation != "":
        # Read the dataset
        path = f"{base}/{folder}/{species}/{simulation}"
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
                label=t("download"),
                data=file,
                file_name=f"{simulation}",
                icon=":material/download:",
            )

        # Setup for reading individual variables in file
        # TODO: allow user to make table from whatever variales they want, turn that into plots
        variable_desc = {}
        for variable in biodata.variables:
            variable_desc[v(variable, "short")] = v(variable, "long")

        st.write(t("create_custom"))
        st.write(t("dataset_vars"))
        st.json(variable_desc, expanded=0)

        # Remove variables that aren't time-related/would be difficult for users to make sense of
        selectable_vars = biodata.variables.copy()
        selectable_vars.pop("ftime")
        selectable_vars.pop("fishing_rate")
        selectable_vars.pop("reprod_rate")
        selectable_vars.pop("rotation_rate")
        selectable_vars.pop("age")
        selectable_vars.pop("mid_bins")
        # Too many things for a user to reasonably be able to see on a plot
        selectable_vars.pop("fish")

        variables = st.multiselect(
            label=t("var_select"),
            options=selectable_vars,
            format_func=lambda x: v(x, "short")
        )

        # TODO: for each selected: get title, get data, if more than 1d do a for loop to add all bins/stock with label
        # Collect data user selected, start with time variable
        selected_data = pd.DataFrame()
        selected_data["year"] = years
        for variable in variables:
            # Get data for the variable
            data = biodata.variables[variable][:].data[100:][:last]

            # If 1D, add data. If 2D, title each bin and add. If 3D, add the bins for each stock then add next stock
            match len(biodata.variables[variable].shape):
                case 1:
                    # Get variable name and units for column title
                    title = f"{v(variable, "short")} ({v(variable, "units")})"

                    # Add data to the dataframe
                    selected_data[title] = data
                case 2:
                    # Some 2D variables have different units/sizes so handle differently
                    match variable:
                        case "stock_size" | "stock_biomass" | "stock_recruit" | "resource":
                            var_name = variable.split("_")[-1]
                            units = v(variable, "units")
                            data = pd.DataFrame(
                                data,
                                columns=[f"{t("stock")} {i + 1} {t(var_name)} ({(units)})" for i in range(data.shape[1])]
                            )
                            selected_data = pd.concat([selected_data, data], axis=1)
                        case "reproduction" | "mortality":
                            data = pd.DataFrame(
                                data,
                                columns=[
                                    f"{t(variable)} {t("biomass")} (kg)",
                                    f"{t(variable)} {t("number")} (#)"
                                ]
                            )
                            selected_data = pd.concat([selected_data, data], axis=1)
                case 3:
                    # TODO: figure out this, gonna take too long so it's a later problem
                    # Some 3D variables have different units/sizes so handle differently
                    continue
                    match variable:
                        case "pop_bins":
                            # TODO: figure out how to smush so it's (years, bins * stocks) with each column being called "bin x stock y population"
                            pass

        selected_data = selected_data.set_index("year")
        st.dataframe(selected_data)

        # Plot once data is selected
        if selected_data.shape[1] > 0:
            figLayout = go.Layout(
                title={
                    "text": t("custom_plot"),
                    "x": 0.5,  # Center title on plot
                    "xanchor": "center",
                },
                # Set labels along with range
                xaxis=dict(
                    title=t("years_x"),
                    range=[0, None]
                ),
                yaxis=dict(
                    title=t("cust_y"),
                    range=[0, None]
                ), 
                template="plotly"  # Default dark theme
            )
            # If ends early, modify title using html to add warning
            if endsEarly:
                figLayout.title["text"] = f"{t("custom_plot")} <br><sup>{t("crash_warning")}</sup>" # type: ignore

            fig = go.Figure(layout=figLayout)

            # Add all data to figure
            for i in range(selected_data.shape[1]):
                trace = go.Scatter(
                    x=years,
                    y=selected_data.iloc[:, i],
                    mode="lines",
                    name=selected_data.columns[i]
                )
                fig.add_trace(trace)

            st.plotly_chart(fig)

        # Plot the simulation (regular plots)
        plots = plot_simulation(path)
        for plot in plots:
            st.plotly_chart(plot)
