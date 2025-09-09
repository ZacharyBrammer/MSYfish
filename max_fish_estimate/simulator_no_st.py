import os

import numpy as np
import pandas as pd

from msy_stocks_rev14_connect import compute_pop_msy


class Simulator:
    def __init__(
        self,
        outdir: str,  # output directory
        fishdata: pd.DataFrame,  # dataframe of species data
        speciesIndex: int,  # index of species in fishdata
    ):
        np.seterr(all="raise")
        outdir = f"max_fish_estimate/"
        self.outdir = outdir

        # grab data from fishdata array
        species = fishdata['scientific'][speciesIndex]
        species = species.split()
        species = species[0] + '_' + species[1]
        self.species = species

        self.iteration = 0  # current iteration number

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

    def to_dict(self, i):
        data = {
            "species": self.species,
            "iteration": i + 1,
            "maxfish": self.maxfish,
            "minrec": self.minrec,
            "asympLen": self.asympLen,
            "growthCoef": self.growthCoef,
            "lenWtPower": self.lenWtPower,
            "lenWtCoef": self.lenWtCoef,
            "maxage": self.maxage,
            "minsize": self.minsize,
            "bgResource": self.bgResource
        }

        return data
