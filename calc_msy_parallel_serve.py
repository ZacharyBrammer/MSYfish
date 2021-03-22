#!/usr/bin/python
#Script to run multiple fisheries model

import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from calc_msy_reprod_serve2 import calc_msy

datadir = '/Users/bwoodson/Documents/UGA/COBIA_LAB/Projects/MSY_Sustainability/msyandpopdynamics/model_output_rev14.nosync/'

if not os.path.exists(datadir):
    os.makedirs(datadir)
    print('Creating data folder: ' + datadir)
    
fishdata=pd.read_excel("/Users/bwoodson/Documents/UGA/COBIA_LAB/Projects/MSY_Sustainability/msyandpopdynamics/fish_growth_data2.xlsx",sheet_name='fish_growth_data')

numspec = fishdata.shape[0]

Parallel(n_jobs=2)(delayed(calc_msy)(datadir,fishdata,kk) for kk in range(0,numspec))



