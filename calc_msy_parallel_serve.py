#!/usr/bin/python
# Script to run multiple fisheries model

import argparse
import os

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from calc_msy_rotational_serve import calc_msy

# command line arguments - will be removed when GUI added
parser = argparse.ArgumentParser(
    prog='MSYParallelServe',
    description='Runs the MSY Fish Model')

files = parser.add_argument_group(title='Paths', description='File paths for inputs/outputs')
files.add_argument(dest='outputdir', help='Path to output files')
files.add_argument(dest='fishdata', help='Path to species growth data file')
files.add_argument('-c', '-connectivity', dest='connectivity', help='Path to optional connectivity file', default=None)

modelVars = parser.add_argument_group(title='Model Variables', description='Variables for running the model')
modelVars.add_argument(dest='startspecies', help='Starting index of species from file', type=int)
modelVars.add_argument(dest='endspecies', help='Ending index of species from file (non-inclusive)', type=int)
modelVars.add_argument(dest='stocks', help='Number of stocks', type=int)
modelVars.add_argument(dest='niter', help='Number of iterations per simulation', type=int)
modelVars.add_argument(dest='years', help='Number of years per simulation', type=int)
modelVars.add_argument(dest='initialPop', help='Initial number of fish', type=int)
modelVars.add_argument(dest='fishing', help='Include fishing in simulation', type=bool)
modelVars.add_argument(dest='rotation', help='Enable rotational closure', type=bool)

args = parser.parse_args()

outputdir = args.outputdir

# create data file if does not exist
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    print('Creating data folder: ' + outputdir)

# read input data file
fishdata = pd.read_excel(args.fishdata)

# determine number of species
numspec = fishdata.shape[0]
start = args.startspecies
end = args.endspecies
if (start < 0 or end > numspec or start > end):
    exit()
print(args._get_args)
# run model - start and end = 14/15

if args.connectivity:
    conn_data = pd.read_excel(args.connectivity)
    connectivity = conn_data.to_numpy()
    print(connectivity)
    connectivity = connectivity[:,1:]
    print(connectivity)
else:
    connectivity = np.array(None)

Parallel(n_jobs=2)(delayed(calc_msy)(outputdir, fishdata, connectivity, kk, args.stocks, args.niter, args.years, args.initialPop, False, args.rotation)
                   for kk in range(start, end))

# Command I used for testing - python calc_msy_parallel_serve.py model_output_new_cons.nosync/ fish_growth_data2.xlsx 14 15 6 6 1 1 false false
