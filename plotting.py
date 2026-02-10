import netCDF4 as nc
import numpy as np
import plotly.graph_objects as go # type: ignore
import streamlit as st

from translate import Translator


def plot_simulation(
    path: str,  # path to simulation output
) -> list[go.Figure]:
    translator = Translator(st.session_state.language)
    t = translator.translate

    # Read the dataset
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

    # List for storing all plots
    plots = []

    # Plot biomass per stock
    # Get per-stock biomass data
    stockBiomass = biodata.variables["stock_biomass"][:].data[100:][:last]
    popDat = {
        t("plot_labels", "biomass_avg"): f"{np.mean(stockBiomass):.2f}",
        t("plot_labels", "biomass_sd"): f"{np.std(stockBiomass):.2f}"
    }

    # Setup layout for interactive figure
    stockBLayout = go.Layout(
        title={
            "text": t("plot_labels", "biomass_time"),
            "x": 0.5,  # Center title on plot
            "xanchor": "center",
        },
        # Set labels along with range
        xaxis=dict(title=t("plot_labels", "years_x"), range=[0, None]),
        yaxis=dict(title=t("plot_labels", "biomass_y"), range=[0, None]),
        template="plotly"  # Default dark theme
    )

    # If ends early, modify title using html to add warning
    if endsEarly:
        stockBLayout.title["text"] = f"{t("plot_labels", "biomass_time")} <br><sup>{t("errors", "crash_warning")}</sup>" # type: ignore

    stockBFig = go.Figure(layout=stockBLayout)

    # Add each stock to figure
    for i in range(stockBiomass.shape[1]):
        trace = go.Scatter(
            x=years,
            y=stockBiomass[:, i],
            mode="lines",
            name=f"{t("sim_settings", "stock_num", n=i+1)}"
        )
        stockBFig.add_trace(trace)

    plots.append(stockBFig)

    # Plot population size over time
    # Get popsize data
    popsize = biodata.variables["popsize"][:].data[100:][:last]

    # Setup layout for interactive figure
    popsizeFigLayout = go.Layout(
        title={
            "text": t("plot_labels", "popsize_time"),
            "x": 0.5,  # Center title on plot
            "xanchor": "center",
        },
        # Set labels along with range
        xaxis=dict(title=t("plot_labels", "years_x"), range=[0, None]),
        yaxis=dict(title=t("plot_labels", "popsize_y"), range=[0, None]),
        template="plotly"  # Default dark theme
    )

    # If ends early, modify title using html to add warning
    if endsEarly:
        popsizeFigLayout.title["text"] = f"{t("plot_labels", "popsize_time")} <br><sup>{t("errors", "crash_warning")}</sup>" # type: ignore

    # Create figure
    popsizeFig = go.Figure(layout=popsizeFigLayout)
    popsizeFig.add_trace(
        go.Scatter(
            x=years,
            y=popsize[:],
            mode="lines"
        )
    )

    plots.append(popsizeFig)

    # Get catch data
    catch = biodata.variables["catch"][:].data[100:][:last]

    # Select just the caught weight data
    catch = catch[:, :, 1]

    # Check if fishing was enabled (there will be non-zeros in array)
    fishing = np.any(catch)

    if fishing:
        # Get average count and weight of catch
        popDat[t("plot_labels", "catch_avg_kg")] = f"{np.mean(catch):.2f}"
        catchCt = biodata.variables["catch"][:].data[100:][:last]
        catchCt = catchCt[:, :, 0]
        popDat[t("plot_labels", "catch_avg_ct")] = f"{np.mean(catchCt):.2f}"

        # Plot catch over time
        # Setup layout for interactive figure
        catchFigLayout = go.Layout(
            title={
                "text": t("plot_labels", "catch_time"),
                "x": 0.5,  # Center title on plot
                "xanchor": "center",
            },
            # Set labels along with range
            xaxis=dict(title=t("plot_labels", "years_x"), range=[0, None]),
            yaxis=dict(title=t("plot_labels", "catch_kg_y"), range=[0, None]),
            template="plotly"  # Default dark theme
        )

        # If ends early, modify title using html to add warning
        if endsEarly:
            catchFigLayout.title["text"] = f"{t("plot_labels", "catch_time")} <br><sup>{t("errors", "crash_warning")}</sup>" # type: ignore

        catchFig = go.Figure(layout=catchFigLayout)

        # Add each stock to figure
        for i in range(catch.shape[1]):
            trace = go.Scatter(
                x=years,
                y=catch[:, i],
                mode="lines",
                name=f"{t("sim_settings", "stock_num", n=i+1)}"
            )
            catchFig.add_trace(trace)

        plots.append(catchFig)

        # Plot per-stock average w/ standard deviations
        # Setup layout for interactive figure
        catchBarLayout = go.Layout(
            title={
                "text": t("plot_labels", "catch_stock"),
                "x": 0.5,  # Center title on plot
                "xanchor": "center",
            },
            # Set labels along with range
            xaxis=dict(title=t("variable", "stock"), range=[0, None]),
            yaxis=dict(title=t("plot_labels", "catch_kg_y"), range=[0, None]),
            template="plotly"  # Default dark theme
        )

        catchBar = go.Figure(layout=catchBarLayout)
        # Get average and std for each stock
        avgs = []
        stds = []
        for i in range(catch.shape[1]):
            avgs.append(np.mean(catch[:, i]))
            stds.append(np.std(catch[:, i]))
            trace = go.Bar(
                y=catch[:, i],
                name=f"{t("sim_settings", "stock_num", n=i+1)}"
            )
            # catchBar.add_trace(trace)
        trace = go.Bar(
            x=[f"{t("sim_settings", "stock_num", n=i+1)}" for i in range(catch.shape[1])],
            y=avgs,
            error_y=dict(
                type="data",
                array=stds,
            )
        )
        catchBar.add_trace(trace)

        plots.append(catchBar)

    st.session_state.sim.popDat = popDat
    # Close file
    biodata.close()

    return plots
