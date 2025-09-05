import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

import pandas as pd
from tqdm import tqdm

from simulator_no_st import Simulator

# --- Load data ---
fishdata = pd.read_excel("fish_growth_data2.xlsx")
speciesList = fishdata["scientific"]

def runSimulation(speciesIndex: int, sim_id: int):
    # --- Generate a high-quality unique seed per simulation ---
    seed_bytes = os.urandom(4) # 32 bits of OS entropy
    seed = int.from_bytes(seed_bytes, "little") + sim_id
    
    # --- Set the global NumPy RNG seed ---
    np.random.seed(seed)

    sim = Simulator("path", fishdata, speciesIndex)
    return sim.to_dict(sim_id)

sims = []

# --- Parallel execution ---
with ProcessPoolExecutor() as executor:
    all_futures = {}
    for species in tqdm(speciesList[:5], desc="Species", leave=True):
        speciesIndex: int = speciesList[speciesList == species].index[0] # type: ignore

        # Submit 10 simulations for this species
        futures = [
            executor.submit(runSimulation, speciesIndex, i)
            for i in range(10)
        ]
        for f in futures:
            all_futures[f] = species  # keep mapping to species

    # One global bar for *all* simulations
    with tqdm(total=len(all_futures), desc="All Sims") as global_bar:
        for f in as_completed(all_futures):
            sims.append(f.result())
            sp = all_futures[f]

            # Update global bar
            global_bar.update(1)

# --- Save results ---
df = pd.DataFrame(sims)
df.to_csv("simulations.csv", index=False)