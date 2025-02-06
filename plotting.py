import os

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def plot_simulation(
        path: str, # path to simulation output
    ):
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

    # Create Legend titles
    stocks = []
    for i in range(len(stockBiomass[0])):
        stocks.append(f"Stock {i + 1}")

    # Create plot
    stockBFig = plt.figure("Stock Biomass vs Time")
    plt.plot(years, stockBiomass)
    plt.title("Biomass Per Stock vs Time")
    plt.xlabel("Time (years)")
    plt.ylabel("Biomass (kg)")
    plt.ylim(bottom=0)
    plt.gca().legend(stocks)

    #plt.savefig("stock_biomass_time")
    #plt.show()

    # Plot catch over time
    # Get catch data
    catch = biodata.variables["catch"][:].data[100:][:last]

    # Select just the caught weight data
    catch = catch[:, :, 1]

    catchFig = plt.figure("Catch Over Time")
    plt.plot(years, catch)
    plt.title("Catch Per Stock vs Time")
    plt.xlabel("Time (years)")
    plt.ylabel("Catch (kg)")
    plt.ylim(bottom=0)
    plt.gca().legend(stocks)
    
    plt.show()

    
    # Close file
    biodata.close()
