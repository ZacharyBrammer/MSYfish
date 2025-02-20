import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import streamlit as st


def plot_simulation(
        path: str, # path to simulation output
    ) -> list[str]:
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

    # Create Legend titles
    stocks = []
    for i in range(len(stockBiomass[0])):
        stocks.append(f"Stock {i + 1}")

    # Create plot
    stockBFig = plt.figure("Stock Biomass vs Time")
    plt.plot(years, stockBiomass)
    if endsEarly:
        plt.suptitle("Biomass Per Stock vs Time")
        plt.title("Warning: population crashed during simulation")
    else:
        plt.title("Biomass Per Stock vs Time")

    plt.xlabel("Time (years)")
    plt.ylabel("Biomass (kg)")
    plt.ylim(bottom=0)
    plt.gca().legend(stocks)

    plt.savefig("plots/stock_biomass_time")

    # Plot catch over time
    # Get catch data
    catch = biodata.variables["catch"][:].data[100:][:last]

    # Select just the caught weight data
    catch = catch[:, :, 1]

    catchFig = plt.figure("Catch Over Time")
    plt.plot(years, catch)
    if endsEarly:
        plt.suptitle("Catch Per Stock vs Time")
        plt.title("Warning: population crashed during simulation")
    else:
        plt.title("Catch Per Stock vs Time")

    plt.xlabel("Time (years)")
    plt.ylabel("Catch (kg)")
    plt.ylim(bottom=0)
    plt.gca().legend(stocks)
    
    plt.savefig("plots/stock_catch_time")

    # Close file
    biodata.close()

    return ["plots/stock_biomass_time.png", "plots/stock_catch_time.png"]
