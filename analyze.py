import os
from typing import Any, Dict, List

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
            compareAll = st.toggle(label="Compare all simulations")
            # Get all simulations in the folder and let user select
            simulations = [""] + os.listdir(f"{base}/{folder}/{species}")
            if not compareAll:
                simulation = st.selectbox(
                    label=t("simulation"),
                    options=simulations
                )
            else:
                simulation = ""
        else:
            simulation = ""
            compareAll = False
    else:
        # If blank folder is selected set species and simulation to blank
        species = simulation = ""
        compareAll = False

    # Once a specific simulation has been selected or compare all is selected
    if simulation != "" or compareAll:
        # Get biodata from either single sim or all sims
        if compareAll:
            # Get the average data for all sims
            average_sims(f"{base}/{folder}/{species}", simulations)
            path = f"{base}/{folder}/{species}/average.nc"
            pass
        else:
            # Default behavior
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
                file_name=f"{simulation}" if simulation != "" else "average.nc",
                icon=":material/download:",
            )

        # Setup for reading individual variables in file
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
                                columns=[
                                    f"{t("stock")} {i + 1} {t(var_name)} ({(units)})" for i in range(data.shape[1])
                                ]
                            )
                            selected_data = pd.concat(
                                [selected_data, data],
                                axis=1
                            )
                        case "reproduction" | "mortality":
                            data = pd.DataFrame(
                                data,
                                columns=[
                                    f"{t(variable)} {t("biomass")} (kg)",
                                    f"{t(variable)} {t("number")} (#)"
                                ]
                            )
                            selected_data = pd.concat(
                                [selected_data, data],
                                axis=1
                            )
                case 3:
                    match variable:
                        case "pop_bins" | "biomass_bins" | "reprod_bins" | "catch_bins" | "mort_bins":
                            var_name = v(variable, "short")
                            units = v(variable, "units")
                            time, bins, stocks = data.shape
                            data = data.reshape(time, bins * stocks)
                            data = pd.DataFrame(
                                data,
                                columns=[
                                    f"{t("bin")} {b + 1} {t("stock")} {s + 1} {var_name} ({(units)})"
                                    for b in range(bins)
                                    for s in range(stocks)
                                ]
                            )
                            selected_data = pd.concat(
                                [selected_data, data],
                                axis=1
                            )
                        case "age_bins":
                            # TODO: Figure out units / if this is right
                            var_name = v(variable, "short")
                            units = v(variable, "units")
                            time, ages, stocks = data.shape
                            data = data.reshape(time, ages * stocks)
                            data = pd.DataFrame(
                                data,
                                columns=[
                                    f"{v("age", "short")} {a + 1} {t("stock")} {s + 1} {var_name} ({(units)})"
                                    for a in range(ages)
                                    for s in range(stocks)
                                ]
                            )
                            selected_data = pd.concat(
                                [selected_data, data],
                                axis=1
                            )
                        case "catch":
                            var_name = v(variable, "short")
                            units = v(variable, "units")
                            time, stocks, vars = data.shape
                            columns = []
                            for i in range(stocks):
                                columns.append(
                                    f"{t("stock")} {i + 1} {var_name} (#)"
                                )
                                columns.append(
                                    f"{t("stock")} {i + 1} {var_name} (kg)"
                                )
                            data = data.reshape(time, stocks * vars)
                            data = pd.DataFrame(
                                data,
                                columns=columns
                            )
                            selected_data = pd.concat(
                                [selected_data, data],
                                axis=1
                            )

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


def average_sims(path: str, simulations: List[str]):
    # Remove the empty string placeholder and previous average file (if applicable)
    sims = simulations.copy()
    sims.remove("")
    if "average.nc" in sims:
        sims.remove("average.nc")

    # Restore path to sim names
    for i in range(len(sims)):
        sims[i] = f"{path}/{sims[i]}"

    refSim = sims[0]

    # Open reference file and set up average file
    with nc.Dataset(refSim) as ref:
        with nc.Dataset(f"{path}/average.nc", "w") as out:
            # Copy dimensions
            for name, dim in ref.dimensions.items():
                out.createDimension(name, len(dim))

            # Copy variables
            for name, var in ref.variables.items():
                out.createVariable(name, var.datatype, var.dimensions)
                # Copy attributes
                out.variables[name].setncatts(
                    {k: var.getncattr(k) for k in var.ncattrs()}
                )

    # Initialize data stacks
    stacks: Dict[str, List[Any]] = {
        name: [] for name in nc.Dataset(refSim).variables.keys()
    }

    # Get all data. If a simulation ended early pad with NaN
    for sim in sims:
        with nc.Dataset(sim) as ds:
            # Determine if simulation ended early
            years = ds.variables["ftime"][:].data[100:]
            endsEarly = np.where(years == 0)[0].size != 0
            if endsEarly:
                last = np.where(years == 0)[0][0]
            else:
                last = None

            # Copy all the data
            for name, var in ds.variables.items():
                data = var[:]

                if "time" in var.dimensions and last is not None:
                    cutoff = 100 + last  # keeps stabilization years in file
                    data[cutoff:] = np.nan

                stacks[name].append(data)

    # Remove fish, age since they have a different shape for every array and users can't see it anyways
    stacks.pop("fish")
    stacks.pop("age")

    # Write averages to file
    with nc.Dataset(f"{path}/average.nc", "a") as out, nc.Dataset(refSim) as ref:
        for name, stack in stacks.items():
            out.variables[name][:] = np.nanmean(
                np.stack(stack, axis=0),
                axis=0
            )
