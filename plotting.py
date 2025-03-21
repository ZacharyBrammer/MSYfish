import netCDF4 as nc
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def plot_simulation(
    path: str,  # path to simulation output
) -> list[go.Figure]:
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

    # Plot biomass per stock
    # Get per-stock biomass data
    stockBiomass = biodata.variables["stock_biomass"][:].data[100:][:last]
    popDat = {
        "Stock Biomass Average": f"{np.mean(stockBiomass):.2f}",
        "Stock Biomass Standard Deviation": f"{np.std(stockBiomass):.2f}"
    }
    st.session_state.popDat = popDat

    # Setup layout for interactive figure
    stockBLayout = go.Layout(
        title={
            "text": "Stock Biomass vs Time",
            "x": 0.5,  # Center title on plot
            "xanchor": "center",
        },
        # Set labels along with range
        xaxis=dict(title="Time (years)", range=[0, None]),
        yaxis=dict(title="Biomass (kg)", range=[0, None]),
        template="plotly"  # Default dark theme
    )

    # If ends early, modify title using html to add warning
    if endsEarly:
        stockBLayout.title["text"] = "Stock Biomass vs Time <br><sup>Warning: population crashed during simulation</sup>"

    stockBFig = go.Figure(layout=stockBLayout)

    # Add each stock to figure
    for i in range(stockBiomass.shape[1]):
        trace = go.Scatter(
            x=years,
            y=stockBiomass[:, i],
            mode="lines",
            name=f"Stock {i+1}"
        )
        stockBFig.add_trace(trace)
    
    # Plot population size over time
    # Get popsize data
    popsize = biodata.variables["popsize"][:].data[100:][:last]

    # Setup layout for interactive figure
    popsizeFigLayout = go.Layout(
        title={
            "text": "Population Size vs Time",
            "x": 0.5,  # Center title on plot
            "xanchor": "center",
        },
        # Set labels along with range
        xaxis=dict(title="Time (years)", range=[0, None]),
        yaxis=dict(title="Population Size", range=[0, None]),
        template="plotly"  # Default dark theme
    )

    # If ends early, modify title using html to add warning
    if endsEarly:
        popsizeFigLayout.title["text"] = "Population Size vs Time <br><sup>Warning: population crashed during simulation</sup>"
    
    # Create figure
    popsizeFig = go.Figure(layout=popsizeFigLayout)
    popsizeFig.add_trace(
        go.Scatter(
            x = years,
            y = popsize[:],
            mode="lines"
        )
    )

    # Plot catch over time
    # Get catch data
    catch = biodata.variables["catch"][:].data[100:][:last]

    # Select just the caught weight data
    catch = catch[:, :, 1]

    # Setup layout for interactive figure
    catchFigLayout = go.Layout(
        title={
            "text": "Catch Per Stock vs Time",
            "x": 0.5,  # Center title on plot
            "xanchor": "center",
        },
        # Set labels along with range
        xaxis=dict(title="Time (years)", range=[0, None]),
        yaxis=dict(title="Catch (kg)", range=[0, None]),
        template="plotly"  # Default dark theme
    )

    # If ends early, modify title using html to add warning
    if endsEarly:
        catchFigLayout.title["text"] = "Catch Per Stock vs Time <br><sup>Warning: population crashed during simulation</sup>"

    catchFig = go.Figure(layout=catchFigLayout)

    # Add each stock to figure
    for i in range(catch.shape[1]):
        trace = go.Scatter(
            x=years,
            y=catch[:, i],
            mode="lines",
            name=f"Stock {i+1}"
        )
        catchFig.add_trace(trace)

    # Close file
    biodata.close()

    return [stockBFig, popsizeFig, catchFig]
