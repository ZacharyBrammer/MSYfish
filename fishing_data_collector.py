import pandas as pd
import time
from tqdm import tqdm

from simulator_no_st import Simulator

fishdata = pd.read_excel("fish_growth_data2.xlsx")
speciesList = fishdata["scientific"]

for species in tqdm(speciesList, desc="All Species"):
    speciesIndex: int = speciesList[speciesList == species].index[0] # type: ignore

    for i in tqdm(range(10), desc=f"Simulating {species}", leave=False):
        sim = Simulator("path", fishdata, speciesIndex)

    break
