import os

import numpy as np
import pandas as pd
import streamlit as st

from msy_stocks_rev14_connect import compute_pop_msy
from translate import Translator

class Simulator:
    def __init__(
            self,
            outdir: str,  # output directory
            fishdata: pd.DataFrame,  # dataframe of species data
            speciesIndex: int,  # index of species in fishdata
        ):
        self.translator = Translator(st.session_state.language)
        self.t = self.translator.translate

        self.outdir = f"simulations/{st.session_state.id}/{outdir}/"

        # grab data from fishdata array
        species = fishdata['scientific'][speciesIndex]
        species = species.split()
        self.species = species[0] + '_' + species[1]
        # TODO: add incrementor to this for the run simulation button
        self.iteration = 0  # current iteration number
        