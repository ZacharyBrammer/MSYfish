import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulator_no_st import Simulator

fishdata = pd.read_excel("fish_growth_data2.xlsx")
speciesList = fishdata["scientific"]

simdata = []

for species in tqdm(speciesList, desc="All Species"):
    speciesIndex: int = speciesList[speciesList == species].index[0] # type: ignore

    for i in tqdm(range(10), desc=f"Simulating {species}", leave=False):
        sim = Simulator("path", fishdata, speciesIndex)
        simdata.append(sim.to_dict(i))

df = pd.DataFrame(simdata)
df.to_csv("simulations.csv")
