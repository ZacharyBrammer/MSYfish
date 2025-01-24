import os

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

def plot_simulation(
        path: str, # path to simulation output
        xAxis: str, # varible from file to plot on x axis
        yAxis: str # variable from file to plot on y axis
    ):
    # Read the dataset
    biodata = nc.Dataset(path, "r")

    # Get x and y axis data
    xData = biodata.variables[xAxis][:].data
    yData = biodata.variables[yAxis][:].data
    
    # Close file
    biodata.close()

    plt.plot(xData, yData)
    plt.show()
