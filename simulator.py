import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import streamlit as st

from msy_stocks_rev14_connect import compute_pop_msy
from translate import Translator


class Simulator:
    def __init__(
        self,
        outdir: str,  # output directory
        fishdata: pd.DataFrame,  # dataframe of species data
        speciesIndex: int,  # index of species in fishdata
        iteration: int = 0, # used for resestting iteration number on new object
    ):
        self.translator = Translator(st.session_state.language)
        self.t = self.translator.translate

        outdir = f"simulations/{st.session_state.id}/{outdir}/"
        self.outdir = outdir

        # grab data from fishdata array
        species = fishdata['scientific'][speciesIndex]
        species = species.split()
        species = species[0] + '_' + species[1]
        self.species = species
        self.speciesIndex = speciesIndex

        self.iteration = iteration  # current iteration number

        # get species parameters and make relevant calculations
        # asymptotic length
        asympLen = np.zeros(2)
        asympLen[0] = fishdata['Lmean'][speciesIndex]
        asympLen[1] = fishdata['Lstd'][speciesIndex]
        self.asympLen = asympLen

        # growth coefficient
        growthCoef = np.zeros(2)
        growthCoef[0] = fishdata['Kmean'][speciesIndex]
        growthCoef[1] = fishdata['Kstd'][speciesIndex]
        self.growthCoef = growthCoef

        # power (b) of length-weight relationship
        lenWtPower = fishdata['bmean'][speciesIndex] + np.random.randn() * fishdata['bstd'][speciesIndex] / 20
        self.lenWtPower = lenWtPower

        # calculate coefficient (a) of length-weight power relationship based on value of lenWtPower
        sl = np.zeros([2])
        sl[0] = fishdata['slope'][speciesIndex]
        sl[1] = fishdata['inter'][speciesIndex]
        lenWtCoef = 10 ** (sl[0] * self.lenWtPower + sl[1])
        self.lenWtCoef = lenWtCoef

        # maximum age
        maxage = int(fishdata['max age'][speciesIndex])
        self.maxage = maxage

        # set reproduction scaling (from earlier sensitivity experiments (see Fig 2)
        reprodmax = -1. * (np.log10(self.maxage / self.growthCoef[0])) - 0.25
        reprodmin = -1. * (np.log10(self.maxage / self.growthCoef[0])) - 4.75

        # number of steps for evaluating recruitment rate
        rstep = 48

        # set recruitment values in log space
        reprodstep = np.logspace(reprodmin, reprodmax, rstep)

        # set asymptotic mass and min size for capture
        winf = (self.lenWtCoef * self.asympLen[0] ** self.lenWtPower) / 1000

        # minimum size caught in kg
        minsize = 0.2 * winf
        self.minsize = minsize

        # set background resource value to scale as winf to constrain run time
        bgResource = np.floor(800 * winf ** 1.2)
        self.bgResource = bgResource

        # set file directories if needed
        if not os.path.exists(outdir + species):
            os.makedirs(outdir + species)

        # set up recruitment estimation variables
        rslap = True
        recruitmentIndex = 0
        fishingRate = np.zeros([1])

        # initilize model by calculating recruitment rate and max fishing rate
        with st.status("Initializing Model", expanded=True) as status:
            with st.status("Estimating Recruitment Rate") as recruitStatus:
                while rslap:
                    if recruitmentIndex < rstep:
                        reprodper = reprodstep[recruitmentIndex]
                        rslap = compute_pop_msy(outdir=outdir, fishingRates=fishingRate, nstocks=1, species=species, asympLen=asympLen, growthCoef=growthCoef, lenWtCoef=lenWtCoef, lenWtPower=lenWtPower, maxage=maxage, minsize=minsize, reprodper=reprodper, backgroundRes=bgResource,
                                                msave=False, iteration=0, btarget=1, rptest=True, environ=False, recruitVar=0.0, conn_matrix=np.array(None), rotation=0, nyr=0, sizes=False, minCatch=0, maxCatch=None, temperature=np.array(None), massChance=None, massMort=None, nfished=0)
                        minrec = 1. * reprodper
                        recruitmentIndex += 1
                    else:
                        minrec = 1. * reprodstep[rstep-1]
                        rslap = False

            self.minrec = minrec
            recruitStatus.update(
                label="Recruitment Rate Calculated",
                state="complete"
            )

            with st.status("Estimating Maximum Fishing Rate") as fishStatus:
                # initialize maximum fishing rate
                maxfish = 0.0

                # estimate maximum fishing rate
                fslap = True
                while fslap:
                    fishingRate = np.zeros([1]) + maxfish
                    fslap = not compute_pop_msy(outdir=outdir, fishingRates=fishingRate, nstocks=1, species=species, asympLen=asympLen, growthCoef=growthCoef, lenWtCoef=lenWtCoef, lenWtPower=lenWtPower, maxage=maxage, minsize=minsize, reprodper=minrec, backgroundRes=bgResource,
                                                msave=False, iteration=0, btarget=1, rptest=True, environ=False, recruitVar=0.5, conn_matrix=np.array(None), rotation=0, nyr=0, sizes=False, minCatch=0, maxCatch=None, temperature=np.array(None), massChance=None, massMort=None, nfished=1)

                    # increment max fishing rate if still viable
                    if fslap:
                        maxfish += 0.01

                self.maxfish = maxfish

                fishStatus.update(
                    label="Max Fishing Rate Found",
                    state="complete"
                )

            status.update(
                label="Model Initialized",
                state="complete",
                expanded=False
            )
        
        self.popDat: Dict[str, str] = {}
        self.plots: List[go.Figure] = []

    def change_outdir(
        self,
        outdir: str # new path for simulation outputs
    ):
        outdir = f"simulations/{st.session_state.id}/{outdir}/"
        self.outdir = outdir

        # Update iteration counter so file names are less confusing
        self.iteration = 0

        # set file directories if needed
        if not os.path.exists(outdir + self.species):
            os.makedirs(outdir + self.species)

    def simulate(
        self,
        connectivity: np.ndarray,  # connectivity matrix, will be None if not included
        stocks: int,  # number of stocks
        years: int,  # number of years per simulation
        fishingRate: float,  # fishing rate as a percentage of biomass
        rotationRate: int,
        sizes: bool,  # enable catch size ranges
        minCatch: float,  # minimum length of fish to catch
        maxCatch: float | None,  # maximum length of fish ot catch
        temperature: np.ndarray,  # temperature of water per year, will be None if disabled
        massChance: float | None,  # yearly chance of a mass mortality event
        massMort: float | None,  # proportion of population to die in mass mortality event
    ):
        # set fishing rate array
        fishingRates = np.full(stocks, fishingRate / 100)

        # run simulation
        compute_pop_msy(outdir=self.outdir, fishingRates=fishingRates, nstocks=stocks, species=self.species, asympLen=self.asympLen, growthCoef=self.growthCoef, lenWtCoef=self.lenWtCoef, lenWtPower=self.lenWtPower, maxage=self.maxage, minsize=self.minsize, reprodper=self.minrec, backgroundRes=self.bgResource,
                        msave=True, iteration=self.iteration, btarget=0, rptest=False, environ=True, recruitVar=0.5, conn_matrix=connectivity, rotation=rotationRate, nyr=years, sizes=sizes, minCatch=minCatch, maxCatch=maxCatch, temperature=temperature, massChance=massChance, massMort=massMort, nfished=stocks)
        self.iteration += 1

    @classmethod
    def load(
        cls,
        sessionId: str
    ) -> list["Simulator"]:
        # Check if file exists for this user, if not return empty list
        path = f"simulations/{sessionId}/sims.pkl"
        if not os.path.exists(path):
            return []
        
        # Rehydrate objects from file
        with open(path, "rb") as f:
            sims = pickle.load(f)
        
        # Restore translator objects
        for sim in sims:
            sim.translator = Translator(st.session_state.language)
        
        return sims

    @classmethod
    def save(
        cls,
        sessionId: str,
        sims: list["Simulator"]
    ):
        path = f"simulations/{sessionId}/"
        os.makedirs(path, exist_ok=True)
        savePath = os.path.join(path, "sims.pkl")

        # Remove translator for pickling
        for sim in sims:
            sim.translator = None # type: ignore
        
        with open(savePath, "wb") as f:
            pickle.dump(sims, f)
