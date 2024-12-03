#!/usr/bin/python
# Script to run multiple fisheries model

import os

import numpy as np
import pandas as pd

from msy_stocks_rev14_connect import compute_pop_msy

# rev10 is most recent and used for data and figures


def calc_msy(
    outdir: str,  # output directory
    fishdata: pd.DataFrame,  # dataframe of species data
    connectivity: np.ndarray,  # connectivity matrix, will be None if not included
    speciesIndex: int,  # index of species in fishdata
    stocks: int,  # number of stocks
    niter: int,  # number of simulations to run
    years: int,  # number of years per simulation
    initialPop: int,  # initial fish population
    fishing: bool,  # enable fishing
    rotation: bool  # enable rotation
):

    minfiles = 1  # minimum files per simulation

    if connectivity.any():
        conn_matrix = connectivity
    else:
        conn_matrix = np.array(None)
        pass

    # grab data from fishdata array
    species = fishdata['species'][speciesIndex]
    species = species.split()
    species = species[0] + '_' + species[1]

    xtest = True  # flag for simulation run
    iteration = 0  # current iteration number
    print('%d ' % speciesIndex + species + ' started...')

    while xtest:
        # set Winf to -1 to initialize
        Winf = -1

        # Asymptotic length
        asympLen = np.zeros(2)
        asympLen[0] = fishdata['Lmean'][speciesIndex]
        asympLen[1] = fishdata['Lstd'][speciesIndex]

        # Growth coefficient
        growthCoef = np.zeros(2)
        growthCoef[0] = fishdata['Kmean'][speciesIndex]
        growthCoef[1] = fishdata['Kstd'][speciesIndex]

        # power (b) of length-weight relationship
        lenWtPower = fishdata['bmean'][speciesIndex] + 0. * \
            np.random.randn()*fishdata['bstd'][speciesIndex]/20

        # calculate coefficient (a) of length-weight power relationship based on value of lenWtPower
        sl = np.zeros([2])
        sl[0] = fishdata['slope'][speciesIndex]
        sl[1] = fishdata['inter'][speciesIndex]
        lenWtCoef = 10**(sl[0]*lenWtPower+sl[1])

        # maximum age
        maxage = int(fishdata['max age'][speciesIndex])

        # set reproduction scaling (from earlier sensitivity experiments (see Fig 2)
        reprodmax = -1.*(np.log10(maxage/growthCoef[0]))-0.25
        reprodmin = -1.*(np.log10(maxage/growthCoef[0]))-4.75

        # set number of steps for evaluating recruitment rate and fishing rate
        rstep = 48
        fstep = 25

        # set recruitment values in log space
        reprodstp = np.logspace(reprodmin, reprodmax, rstep)

        # set asymptotic mass and min size for capture
        Winf = (lenWtCoef*asympLen[0]**lenWtPower)/1000
        # minimum size caught in kg
        minsize = 0.2*Winf

        # set background resource value to scale as Winf to constrain run time
        R = np.floor(400*Winf**1.2)

        # set file directories if needed
        if not (np.isnan(asympLen[0]) or np.isnan(growthCoef[0]) or np.isnan(lenWtPower) or np.isnan(maxage)):
            if not os.path.exists(outdir + species) and iteration == 0:
                os.makedirs(outdir + species)
                print('Creating data folder: ' + 'model_output/' + species)
            elif iteration == 0:
                filelist = [f for f in os.listdir(
                    outdir + species) if f.endswith('.nc')]
                for f in filelist:
                    os.remove(os.path.join(outdir + species, f))

            # recruitment estimate flag
            rslap = True

            # set index for recruitment loop
            recruitmentIndex = 0

            # set fishing rate to zero for recruitment loop - per stock
            fishingRates = np.zeros([stocks])

            # estimate minimum viable recruitment rate
            while rslap:
                if recruitmentIndex < rstep:
                    reprodper = reprodstp[recruitmentIndex]
                    fishingRates = np.zeros([stocks])
                    rslap = compute_pop_msy(outdir, fishingRates, stocks, stocks, species, asympLen, growthCoef, lenWtCoef, lenWtPower,
                                            maxage, minsize, reprodper, R, False, iteration, 1, True, False, 0.0, conn_matrix, 0, years)
                    minrec = 1.*reprodper
                    recruitmentIndex = recruitmentIndex + 1
                else:
                    minrec = 1.*reprodstp[rstep-1]
                    rslap = False

            minrec = 1.2*minrec

            # initialize maximum fishing rate
            maxfish = 0.0

            # estimate maximum fishing rate
            while fishing:
                fishingRates = np.zeros([stocks])+maxfish
                fishing = compute_pop_msy(outdir, fishingRates, stocks, stocks, species, asympLen, growthCoef, lenWtCoef,
                                          lenWtPower, maxage, minsize, minrec, R, False, iteration, 1, False, False, .5, conn_matrix, 0, years)
                maxfish = maxfish + .01

            # set index for main model run over various fishing rates
            stocktest = True

            # set fishing rate vector based on maximum fishing rate and steps
            mfstp = np.linspace(0, maxfish, fstep)
            mfstp[0] = 0

            rotationArray = np.arange(0, fstep)
            rotationArray = np.append(rotationArray, 500)

            # set number of stocks to be fished initially to one
            nfish = initialPop

            # print('max fishing rate finished...rate = ',maxfish)

            # set parameters
            rvar = 0.5
            msave = True
            btarget = 0
            rptest = True
            environ = True

            # perform model simulations looping over number of fished stockes and fishing rate
            while stocktest:

                for ii in range(0, fstep+1):  # 41 for complete runs

                    fishingRates[0:nfish] = 0.1
                    slap = compute_pop_msy(outdir, fishingRates, stocks, nfish, species, asympLen, growthCoef, lenWtCoef, lenWtPower, maxage, minsize, minrec,
                                           R, msave, iteration, btarget, rptest, environ, rvar, conn_matrix, rotationArray[ii], years)

                stocklist = [g for g in os.listdir(outdir + species) if g.endswith(
                    '%d' % nfish, 19, 20) and g.endswith('_' + '%d' % iteration + '.nc')]
                stock_files = len(stocklist)

                # check to see if minimum files for each simulation is reached to estimate surplus prodction curves
                if stock_files >= minfiles:
                    nfish = nfish+1
                    if nfish > stocks:
                        stocktest = False
                else:
                    stocktest = False

            # check to see if number of files is sufficient based on minfiles and number of stocks, if reached proceed to next iteration, if not restart
            datalist = [f for f in os.listdir(
                outdir + species) if f.endswith('_' + '%d' % iteration + '.nc')]
            number_files = len(datalist)

            if number_files < minfiles:
                for f in datalist:
                    os.remove(os.path.join(outdir + species, f))
            else:
                iteration = iteration+1
                if iteration >= (niter):
                    xtest = False

        else:  # flag for data not sufficient and exit
            print('Insufficient Data...not running ' + species)
            xtest = False

    print('%d ' % speciesIndex + species + ' is done.')
    return True
