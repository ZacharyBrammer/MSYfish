import os

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

def plot_simulation(
        path: str, # path to simulation output
    ):
    # Read the dataset
    biodata = nc.Dataset(path, "r")

    # Subtract 100 from all years to remove stabilization period
    years = biodata.variables["ftime"][:].data[100:]
    years -= 100

    # Plot biomass per stock
    # Get per-stock biomass data
    stockBiomass = biodata.variables["stock_biomass"][:].data[100:]

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
    catch = biodata.variables["catch"][:].data[100:]

    # Select just the caught # data
    catch = catch[:, :, 0]

    catchFig = plt.figure("Catch Over Time")
    plt.plot(years, catch)
    
    plt.show()

    
    # Close file
    biodata.close()

