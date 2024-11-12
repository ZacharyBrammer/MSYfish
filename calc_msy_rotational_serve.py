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
    initialpop: int,  # initial fish population
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
    print(type(species))

    xtest = True  # flag for simulation run
    iteration = 0  # current iteration number
    print('%d ' % speciesIndex + species + ' started...')
    runs = 0

    while xtest:
        runs += 1
        
        # set Winf to -1 to initialize
        Winf = -1

        # Asymptotic length
        asympLen = np.zeros(2)
        asympLen[0] = fishdata['Lmean'][speciesIndex]
        asympLen[1] = fishdata['Lstd'][speciesIndex]
        print(asympLen)

        # Growth coefficient
        growthCoef = np.zeros(2)
        growthCoef[0] = fishdata['Kmean'][speciesIndex]
        growthCoef[1] = fishdata['Kstd'][speciesIndex]
        print(growthCoef)

        # length-weight power relationship
        b = fishdata['bmean'][speciesIndex] + 0. * \
            np.random.randn()*fishdata['bstd'][speciesIndex]/20
        print(b)

        # calculate coefficient (a) of length-weight power relationship based on value of b
        sl = np.zeros([2])
        sl[0] = fishdata['slope'][speciesIndex]
        sl[1] = fishdata['inter'][speciesIndex]
        print(sl)
        a = 10**(sl[0]*b+sl[1])
        print(a)

        # maximum age
        maxage = int(fishdata['max age'][speciesIndex])
        print(maxage)

        # set reproduction scaling (from earlier sensitivity experiments (see Fig 2)
        reprodmax = -1.*(np.log10(maxage/growthCoef[0]))-0.25
        reprodmin = -1.*(np.log10(maxage/growthCoef[0]))-4.75
        print(reprodmax)
        print(reprodmin)

        # set number of steps for evaluating recruitment rate and fishing rate
        rstep = 48
        fstep = 25

        # set recruitment values in log space
        reprodstp = np.logspace(reprodmin, reprodmax, rstep)
        print(reprodstp)

        # set biomass target for each level of fstep for surplus production curves
        btarget = np.linspace(1, 0, fstep)
        print(btarget)

        # set asymptotic mass and min size for capture
        Winf = (a*asympLen[0]**b)/1000
        print(Winf)
        # minimum size caught in kg
        minsize = 0.2*Winf
        print(minsize)

        # set background resource value to scale as Winf to constrain run time
        R = np.floor(400*Winf**1.2)
        print(R)

        # set file directories if needed
        if not (np.isnan(asympLen[0]) or np.isnan(growthCoef[0]) or np.isnan(b) or np.isnan(maxage)):
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

            # set fishing rate to zero for recruitment loop
            f0 = np.zeros([stocks])
            print(f0)

            # estimate minimum viable recruitment rate
            while rslap:
                if recruitmentIndex < rstep:
                    reprodper = reprodstp[recruitmentIndex]
                    f0 = np.zeros([stocks])
                    rslap = compute_pop_msy(outdir, f0, stocks, stocks, species, asympLen, growthCoef, a, b,
                                            maxage, minsize, reprodper, R, 0, iteration, 1, 1, 0, 0, 0.0, 0, conn_matrix, 0)
                    minrec = 1.*reprodper
                    recruitmentIndex = recruitmentIndex + 1
                    print(minrec)
                    print(rslap)
                else:
                    minrec = 1.*reprodstp[rstep-1]
                    rslap = False

            minrec = 1.2*minrec

            # initialize maximum fishing rate
            maxfish = 0.0

            # estimate maximum fishing rate
            while fishing:
                print(True)
                f0 = np.zeros([stocks])+maxfish
                fishing = compute_pop_msy(outdir, f0, stocks, stocks, species, asympLen, growthCoef, a,
                                        b, maxage, minsize, minrec, R, 0, iteration, 1, 0, 0, 0, .5, 0, conn_matrix, 0)
                maxfish = maxfish + .01

            # set index for main model run over various fishing rates
            stocktest = True

            # set fishing rate vector based on maximum fishing rate and steps
            mfstp = np.linspace(0, maxfish, fstep)
            mfstp[0] = 0

            rotationArray = np.arange(0, fstep)
            rotationArray = np.append(rotationArray, 500)
            print(rotationArray)

            # set number of stocks to be fished initially to one
            nfish = stocks
            print(nfish)

            # print('max fishing rate finished...rate = ',maxfish)

            # set parameters
            rvar = 0.5
            msave = 1
            btarget = 0
            rptest = 0
            environ = 1
            ft = 0

            # perform model simulations looping over number of fished stockes and fishing rate
            while stocktest:

                for ii in range(0, fstep+1):  # 41 for complete runs

                    f0[0:nfish] = 0.1
                    print(f0)
                    print(fstep)
                    slap = compute_pop_msy(outdir, f0, stocks, nfish, species, asympLen, growthCoef, a, b, maxage, minsize, minrec,
                                           R, msave, iteration, btarget, rptest, environ, ft, rvar, connectivity, conn_matrix, rotationArray[ii])
                    print(slap)
                

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
        print(runs)

    print('%d ' % speciesIndex + species + ' is done.')
