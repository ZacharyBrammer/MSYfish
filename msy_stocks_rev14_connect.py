#!/usr/bin/python
# Individual-based population dynamics model
# 20 Nov 21 - added fishing adjustment to insure B/K levels off

import netCDF4 as nc
import numpy as np


def compute_pop_msy(
    outdir: str,  # output directory
    fishingRates: np.ndarray,  # array with fishing rate per stock
    nstocks: int,  # number of stocks
    nfish: int,  # number of fish
    species: str,  # name of species
    asympLen: np.ndarray,  # asymptotic length array
    growthCoef: np.ndarray,  # growth coefficient array
    lenWtCoef: float,  # length-weight relationship coefficient (a)
    lenWtPower: float,  # length-weight power relationship (b)
    maxage: int,  # maximum age for species
    minsize: float,  # minimum weight that can be caught
    reprodper: float,  # percentage of reproduction that turns into new recruits
    backgroundRes: float,  # background resource value
    msave: bool,  # flag for saving output
    iteration: int,  # current iteration number
    btarget: float,  # biomass target
    rptest: bool,  # flag for simulation length when testing recruitment and population sustainability
    environ: bool,  # enable environmental scaling
    recruitVar: float,  # variability in recruitment
    # connectivity matrix - to disable connectivity pass np.array(None)
    conn_matrix: np.ndarray,
    rotation: int,  # rotation rate
    nyr: int # number of years
) -> bool:
    outfile = outdir + species + '/msy_stocks_' + '%d' % fishingRates.size + '_nfish_' + '%d' % nfish + '_mfish_' + \
        '%.4f' % np.max(fishingRates) + '_rot_' + '%03d' % rotation + '_' + \
        '%d' % iteration + '.nc'  # '_rec_' + '%.4f' % reprodper

    # initial conditions
    resourceAvail = backgroundRes/nstocks  # resource availability
    if environ:
        env = 0.05  # enviromental scaling effect
        rho = 0.25  # autocorrelation factor
    else:
        env = 0
        rho = 0

    # if connectivity matrix provided, enable connectivity
    connectivity = conn_matrix.any()

    # rotational closure parameters
    RM = 0

    initialPop = int(400)  # initial population size
    slap = False  # restart flag

    delt = 1  # timestep in years
    #nyr = 300  # length of simulation in years
    if rptest:
        nyr = 5 * maxage
        if nyr < 200:
            nyr = 300

    tlength = int(np.ceil(nyr/delt))
    yy = np.zeros([tlength, 1])

    # reprodper = .016 #percentage of reproduction that turns into new recruits (.03-.06)
    scf = 2
    nbins = 24

    # fish species growth parameters
    fish = np.zeros([tlength, initialPop])
    stock = np.random.randint(0, nstocks, size=initialPop)
    age = np.random.randint(1, maxage, size=initialPop)
    sex = np.random.randint(0, 2, size=initialPop)  # 0 - male 1 - female
    initialAge = 0.25
    maturity = np.zeros([initialPop])

    gvar = growthCoef[1] * np.random.randn(initialPop) / 2
    gvar[gvar <= -growthCoef[1]] = 0
    initialAsympLen = asympLen[0] + asympLen[1] * np.random.randn(initialPop)
    initialAsympLen[initialAsympLen < 0] = asympLen[0]

    # asymptotic weight based on asympWt=asympLen^lenWtPower
    asympWt = (lenWtCoef*initialAsympLen**lenWtPower)/1000
    wtMat = compute_wtMat(asympLen[0], growthCoef[0],
                          lenWtCoef, lenWtPower, maxage)
    initialWt = asympWt * (1 - np.exp(-(growthCoef[0] + gvar) * initialAge)) ** lenWtPower

    binMin = 0.1 * (lenWtCoef * asympLen[0] ** lenWtPower) / 1000 * (1 - np.exp(-(growthCoef[0]) * initialAge)) ** lenWtPower
    binMax = 10 * (lenWtCoef * asympLen[0] ** lenWtPower) / 1000

    fish[0, :] = asympWt * (1-np.exp(-(growthCoef[0] + gvar) * age)) ** lenWtPower

    m00 = 0.95

    # scaled so that age 1 mortality is m00
    m0 = m00*np.mean(initialWt)**(0.25)

    newgrowth = np.zeros([tlength])
    reprodOut = np.zeros([tlength])
    reproduction = np.zeros([tlength, 2])
    preyconsumed = np.zeros([tlength])

    resource = np.zeros([tlength, nstocks])
    eps = np.zeros([tlength])

    fishWt = np.zeros([tlength])
    catch = np.zeros([tlength, nstocks, 2])
    mortality = np.zeros([tlength, 2])
    popsize = np.zeros([tlength, 1])
    biomass = np.zeros([tlength, 1])
    sexRatio = np.zeros([tlength, 1])
    stockSize = np.zeros([tlength, nstocks])
    stockBiomass = np.zeros([tlength, nstocks])
    stockRec = np.zeros([tlength, nstocks])
    reprodTot = np.zeros([tlength, nstocks])

    popsize[0] = initialPop
    biomass[0] = sum(fish[0, :])

    for kk in range(0, nstocks):
        stockSize[0, kk] = fish[0, stock == kk].size
        stockBiomass[0, kk] = np.sum(fish[0, stock == kk])
        stockRec[0, kk] = 0

    mortBin = np.zeros([tlength, nbins, nstocks])
    popBin = np.zeros([tlength, nbins, nstocks])
    biomassBin = np.zeros([tlength, nbins, nstocks])
    ageBin = np.zeros([tlength, maxage, nstocks])
    catchBin = np.zeros([tlength, nbins, nstocks])
    reprodBin = np.zeros([tlength, nbins, nstocks])

    [nn, numfish] = fish.shape

    # print ('Starting model run...')
    for ii in range(1, nyr):
        if ii < 100:
            ff = np.zeros([nstocks])
            mnpop = 1
        # rotational closure
        elif ii >= 100 and rotation > 0:
            ff = fishingRates[:] * nstocks / (nstocks - 1)
            if ii > 0 and np.remainder(ii, rotation) == 0:
                RM += 1
                if RM > (nstocks - 1):
                    RM = 0
                ff[RM] = 0
            else:
                ff[RM] = 0
        else:
            ff = fishingRates
            mnpop = np.mean(popsize[ii-5:ii]) / np.mean(popsize[40:50])

        if mnpop < btarget:
            ff = btarget * ff

        netgrowth = np.zeros([numfish])
        consumed = np.zeros([numfish])
        reprod = np.zeros([numfish])
        eps[ii] = rho * eps[ii-1] + np.sqrt(1 - rho ** 2) * env * resourceAvail * np.random.normal()
        resource[ii, :] = resourceAvail+eps[ii]

        dead = np.zeros([numfish])
        fished = np.zeros([numfish])

        fishWt[ii] = 2 * np.mean(fish[ii-1, :])
        order = np.random.permutation(numfish)

        for kk in range(0, numfish):

            jj = order[kk]
            if dead[jj] == 0:
                bioOrder = np.random.permutation(3)

                for mm in bioOrder:
                    if mm == 0:
                        # individual consumption of resource
                        consumption = 6.0 * fish[ii-1, jj] ** 1.00
                        production = 3.0 * fish[ii-1, jj] ** 0.75
                        metabolism = consumption-production

                        if resource[ii, stock[jj]] >= consumption:
                            availableEnergy = consumption
                        elif fish[ii-1, jj] > 100 * np.mean(initialWt):
                            availableEnergy = resource[ii, stock[jj]]
                            cann = np.abs(
                                (fish[ii-1, :] > 0) * (fish[ii-1, :] <= .01 * fish[ii-1, jj]) * (1 - dead))
                            idcan = np.asarray(cann.nonzero())
                            [dd, ee] = idcan.shape
                            for cc in range(0, ee):
                                eatid = idcan[0, cc]
                                cannibal = np.random.random()
                                cann_rate = 1.0 * fish[ii-1, eatid] ** (-0.25)
                                cann_prob = 1 - np.exp(-cann_rate * delt)
                                if cannibal <= cann_prob and availableEnergy < consumption:
                                    availableEnergy = availableEnergy + fish[ii-1, eatid]
                                    dead[eatid] = 1
                        else:
                            consumption = resource[ii, stock[jj]]
                            availableEnergy = resource[ii, stock[jj]]

                        consumed[jj] = consumption

                       # growth and reproduction
                        if (fish[ii, jj] < wtMat and availableEnergy < metabolism):
                            mortality[ii, 0] += 1
                            mortality[ii, 1] += fish[ii-1, jj]
                            dead[jj] = 1
                            consumption = 0
                        elif availableEnergy < metabolism:
                            growth = 0
                            reprod[jj] = 0
                            netgrowth[jj] = availableEnergy - metabolism
                        else:
                            pconsum = (availableEnergy - metabolism) / production
                            growth = pconsum * asympWt[jj] * lenWtPower * (growthCoef[0] + gvar[jj]) * (fish[ii-1, jj] / asympWt[jj]) ** (
                                1 - 1 / lenWtPower) * (1 - (fish[ii-1, jj] / asympWt[jj]) ** (1 / lenWtPower))
                            netgrowth[jj] = growth*delt
                            if maturity[jj] == 1 and sex[jj] == 1:
                                eggMass = (.008 *
                                           fish[ii-1, jj] ** .109) / 1000 ** 2
                                reprod[jj] = int(
                                    (availableEnergy - growth) / eggMass)
                                if reprod[jj] < 0:
                                    reprod[jj] = 0

                        fish[ii, jj] = fish[ii-1, jj] + netgrowth[jj]

                    if mm == 1:
                        # natural mortality (including predation)
                        mort = m0 * fish[ii-1, jj] ** (-0.25) + 1 / (1 + np.exp(-0.6 * (age[jj] - maxage)))

                        morprob = 1 - np.exp(-mort * delt)

                        death = np.random.random()

                        if death < morprob:
                            mortality[ii, 0] += 1
                            mortality[ii, 1] += fish[ii-1, jj]
                            dead[jj] = 1
                            consumption = 0

                    if mm == 2:
                        # fishing mortality
                        caught = np.random.random()

                        if ff[stock[jj]] > 0:
                            fishmort = scf * ff[stock[jj]] / (1 + np.exp(-0.05 * (fish[ii-1, jj] - fishWt[ii])))
                            fishmprob = 1 - np.exp(-fishmort * delt)
                        else:
                            fishmort = 0
                            fishmprob = 0

                        if caught < fishmprob and fish[ii, jj] > minsize:
                            fished[jj] = 1
                            dead[jj] = 1
                            catch[ii, stock[jj], 0] += 1
                            catch[ii, stock[jj], 1] += fish[ii, jj]
                            consumption = 0

                age[jj] = age[jj] + 1

                # check if fish reaches maturity
                if maturity[jj] == 0:
                    pmat = 1 / (1 + np.exp(-2 * (fish[ii, jj] - wtMat)))
                    ptest = np.random.randn()
                    if ptest > pmat:
                        maturity[jj] = 1

                resource[ii, stock[jj]] -= consumption
                if resource[ii, stock[jj]] < 0:
                    resource[ii, stock[jj]] = 0

        # update mass and age data for dead, caught, or new individuals
        idx = np.asarray(dead).nonzero()
        tempfish = fish[ii, idx]
        nsx = stock[idx]

        binRanges = np.logspace(np.log10(binMin), np.log10(binMax), nbins+1)

        for kk in range(0, nstocks):
            if nsx[nsx == kk].size > 0:
                [mortBin[ii, :, kk], binEdges] = np.histogram(tempfish[0, nsx == kk], bins=binRanges)  # log

        idf = np.asarray(fished).nonzero()
        nst = stock[idf]
        tempfish = fish[ii, idf]

        fish = np.delete(fish, idx, axis=1)
        age = np.delete(age, idx)
        sex = np.delete(sex, idx)
        stock = np.delete(stock, idx)
        reprod = np.delete(reprod, idx)
        maturity = np.delete(maturity, idx)
        asympWt = np.delete(asympWt, idx)
        initialWt = np.delete(initialWt, idx)
        gvar = np.delete(gvar, idx)

        for kk in range(0, nstocks):
            if nst[nst == kk].size > 0:
                [catchBin[ii, :, kk], binEdges] = np.histogram(tempfish[0, nst == kk], bins=binRanges)  # log

        for kk in range(0, nstocks):
            for ll in range(0, nbins):
                ids = np.where(np.logical_and(np.logical_and(fish[ii-1, :] >= binEdges[ll], fish[ii-1, :] < binEdges[ll+1]), stock == kk))
                if ids[0].shape[0] > 0:
                    biomassBin[ii, ll, kk] = np.sum(fish[ii-1, ids])
                    reprodBin[ii, ll, kk] = np.sum(reprod[ids])
                    mid_bins = binEdges[0:-1] + (binEdges[1] - binEdges[0]) / 2

        for kk in range(0, nstocks):
            [popBin[ii, :, kk], _] = np.histogram(fish[ii, stock == kk], bins=binRanges)  # log
            [ageBin[ii, :, kk], _] = np.histogram(age[stock == kk], bins=maxage, range=(0.5, maxage+0.5))

        [nn, numfish] = fish.shape
        
        # no more fish, write file
        if (numfish == 0):
            slap = True
            break

        # update mass and age data for new recruits - make user input, default .5
        recruitVar = 0.5

        if connectivity:
            reproductionMatrix = np.zeros([nstocks, nstocks])
            numrecStock = np.zeros([nstocks, 1])

            for kk in range(0, nstocks):
                reprodTot[ii, kk] = np.sum(reprod[stock == kk])
                for mm in range(0, nstocks):
                    reproductionMatrix[kk, mm] = conn_matrix[kk, mm] * reprodTot[ii, kk]
                sigrec = np.random.lognormal(0., recruitVar)
                numrecStock[kk] = int(sigrec * np.sum(reprodper / 1e5 * reproductionMatrix[:, kk]))

            numrec = int(np.sum(numrecStock))
        else:
            sigrec = np.random.lognormal(0., recruitVar)
            numrec = int(sigrec*reprodper / 1e5 * np.sum(reprod))

        if numrec > 0:
            if connectivity:
                probStock = np.cumsum(numrecStock / numrec)
                stockn = np.random.random(size=numrec)
                newstock = np.zeros([numrec], dtype=int)

                for rr in range(0, nstocks):
                    idr = np.asarray(np.nonzero(stockn <= probStock[rr]))
                    stockn[idr] = 2
                    newstock[idr] = int(rr)
            else:
                newstock = np.random.randint(0, nstocks, size=numrec)

            reproduction[ii, 0] = numrec
            newrec = np.zeros([nn, numrec])
            newgvar = growthCoef[1] * np.random.randn(numrec)
            newgvar[newgvar <= -growthCoef[0]] = 0
            newage = np.zeros([numrec])
            newsex = np.random.randint(0, 2, size=numrec)
            newAsympLen = asympLen[0] + asympLen[1] * np.random.randn(numrec)
            newAsympLen[newAsympLen < 0] = asympLen[0]
            newAsympWt = (lenWtCoef * newAsympLen ** lenWtPower) / 1000
            newInitialWt = newAsympWt * (1 - np.exp(-(growthCoef[0] + newgvar) * initialAge)) ** lenWtPower
            newrec[ii, :] = newInitialWt
            reproduction[ii, 1] = np.sum(newrec[ii, :])

            fish = np.append(fish, newrec, axis=1)
            age = np.append(age, newage)
            sex = np.append(sex, newsex)
            stock = np.append(stock, newstock)
            maturity = np.append(maturity, np.zeros([numrec]))
            asympWt = np.append(asympWt, newAsympWt)
            initialWt = np.append(initialWt, newInitialWt)
            gvar = np.append(gvar, newgvar)

        [nn, numfish] = fish.shape

        # calculate population parameters for data storage
        preyconsumed[ii] = np.sum(consumed)
        newgrowth[ii] = np.sum(netgrowth)
        reprodOut[ii] = np.sum(reprod)
        yy[ii] = ii

        if numrec > 0:
            popsize[ii] = numfish-numrec
            biomass[ii] = np.sum(fish[ii, :]) - np.sum(newrec[ii, :])
        else:
            popsize[ii] = numfish
            biomass[ii] = np.sum(fish[ii, :])

        sexRatio[ii] = np.sum(sex)/sex.shape[0]

        for kk in range(0, nstocks):
            stockSize[ii, kk] = fish[ii, stock == kk].size
            stockBiomass[ii, kk] = np.sum(fish[ii, stock == kk])
            
            if numrec > 0:
                stockRec[ii, kk] = newrec[ii, newstock == kk].size

    # create output file
    if msave:
        biodata = create_popdynnc(
            outfile, nyr, numfish, nbins, maxage, nstocks)

        # write model output to
        biodata.variables['ftime'][:] = yy
        biodata.variables['fishing_rate'][:] = fishingRates
        biodata.variables['reprod_rate'][:] = reprodper
        biodata.variables['fish'][:, :] = fish
        biodata.variables['popsize'][:] = popsize
        biodata.variables['biomass'][:] = biomass
        biodata.variables['consumption'][:] = preyconsumed
        biodata.variables['newgrowth'][:] = newgrowth
        biodata.variables['reprod_out'][:] = reprodOut
        biodata.variables['reproduction'][:, :] = reproduction
        biodata.variables['catch'][:, :, :] = catch
        biodata.variables['age'][:] = age
        biodata.variables['wfish'][:] = fishWt
        biodata.variables['resource'][:, :] = resource
        biodata.variables['mortality'][:, :] = mortality
        biodata.variables['stock_size'][:, :] = stockSize
        biodata.variables['stock_biomass'][:, :] = stockBiomass
        biodata.variables['stock_recruit'][:, :] = stockRec
        biodata.variables['sex_ratio'][:] = sexRatio

        biodata.variables['mort_bins'][:, :, :] = mortBin
        biodata.variables['catch_bins'][:, :, :] = catchBin
        biodata.variables['pop_bins'][:, :, :] = popBin
        biodata.variables['biomass_bins'][:, :, :] = biomassBin
        biodata.variables['reprod_bins'][:, :, :] = reprodBin
        biodata.variables['age_bins'][:, :, :] = ageBin
        biodata.variables['mid_bins'][:] = mid_bins
        biodata.variables['rotation_rate'][:] = rotation

        biodata.close()

    return slap


# Subroutine to create output netcdf file
def create_popdynnc(outfile, tlength, numpop, nbins, maxage, nstocks):
    biodata = nc.Dataset(outfile, 'w')
    biodata.createDimension('time', tlength)
    biodata.createDimension('numpop', numpop)
    biodata.createDimension('numbins', nbins)
    biodata.createDimension('maxage', maxage)
    biodata.createDimension('nstocks', nstocks)
    biodata.createDimension('ones', 1)
    biodata.createDimension('twos', 2)

    # Time
    biodata.createVariable('ftime', 'f8', ('time'))
    biodata.variables['ftime'].long_name = 'time since model start'
    biodata.variables['ftime'].units = 'years'

    # Fish Rate
    biodata.createVariable('fishing_rate', 'f8', ('nstocks'))
    biodata.variables['fishing_rate'].long_name = 'fishing rate in each stock'
    biodata.variables['fishing_rate'].units = '%'

    # Reproduction Rate
    biodata.createVariable('reprod_rate', 'f8', ('nstocks'))
    biodata.variables['reprod_rate'].long_name = 'mean survivial rate of larvae'
    biodata.variables['reprod_rate'].units = '%'

    # Rotation Rate
    biodata.createVariable('rotation_rate', 'f8', ('ones'))
    biodata.variables['rotation_rate'].long_name = 'period of time before reserve rotation'
    biodata.variables['rotation_rate'].units = 'years'

    # Fish Size
    biodata.createVariable('fish', 'f8', ('time', 'numpop'))
    biodata.variables['fish'].long_name = 'individual size through time'
    biodata.variables['fish'].units = 'kg'

    # Population Size
    biodata.createVariable('popsize', 'f8', ('time'))
    biodata.variables['popsize'].long_name = 'population size'
    biodata.variables['popsize'].units = '# of individuals'

    biodata.createVariable('pop_bins', 'f8', ('time', 'numbins', 'nstocks'))
    biodata.variables['pop_bins'].long_name = 'populations size in size bins'
    biodata.variables['pop_bins'].units = '#'

    # Sex Ratio
    biodata.createVariable('sex_ratio', 'f8', ('time'))
    biodata.variables['sex_ratio'].long_name = 'proportion female'
    biodata.variables['sex_ratio'].units = 'proportion'
    # Stock Size
    biodata.createVariable('stock_size', 'f8', ('time', 'nstocks'))
    biodata.variables['stock_size'].long_name = 'stock size'
    biodata.variables['stock_size'].units = '# of individuals'

    biodata.createVariable('stock_biomass', 'f8', ('time', 'nstocks'))
    biodata.variables['stock_biomass'].long_name = 'stock biomass'
    biodata.variables['stock_biomass'].units = '# of individuals'

    biodata.createVariable('stock_recruit', 'f8', ('time', 'nstocks'))
    biodata.variables['stock_recruit'].long_name = 'stock recruits'
    biodata.variables['stock_recruit'].units = '# of individuals'

    # Biomass
    biodata.createVariable('biomass', 'f8', ('time'))
    biodata.variables['biomass'].long_name = 'biomass'
    biodata.variables['biomass'].units = 'kg'

    biodata.createVariable('biomass_bins', 'f8',
                           ('time', 'numbins', 'nstocks'))
    biodata.variables['biomass_bins'].long_name = 'biomass in size bins'
    biodata.variables['biomass_bins'].units = '#'

    # New growth
    biodata.createVariable('newgrowth', 'f8', ('time'))
    biodata.variables['newgrowth'].long_name = 'new biomass'
    biodata.variables['newgrowth'].units = 'kg'

    # Reproductive Output
    biodata.createVariable('reprod_out', 'f8', ('time'))
    biodata.variables['reprod_out'].long_name = 'reproductive output'
    biodata.variables['reprod_out'].units = 'number of eggs'

    # Consumption
    biodata.createVariable('consumption', 'f8', ('time'))
    biodata.variables['consumption'].long_name = 'new biomass'
    biodata.variables['consumption'].units = 'kg'

    # Reproduction
    biodata.createVariable('reproduction', 'f8', ('time', 'twos'))
    biodata.variables['reproduction'].long_name = 'biomass and number'
    biodata.variables['reproduction'].units = 'kg and #'

    biodata.createVariable('reprod_bins', 'f8', ('time', 'numbins', 'nstocks'))
    biodata.variables['reprod_bins'].long_name = 'reproduction in size bins'
    biodata.variables['reprod_bins'].units = '#'

    # Catch
    biodata.createVariable('catch', 'f8', ('time', 'nstocks', 'twos'))
    biodata.variables['catch'].long_name = 'catch biomass and numbers'
    biodata.variables['catch'].units = 'kg and #'

    biodata.createVariable('catch_bins', 'f8', ('time', 'numbins', 'nstocks'))
    biodata.variables['catch_bins'].long_name = 'catch in size bins'
    biodata.variables['catch_bins'].units = '#'

    # Age
    biodata.createVariable('age', 'f8', ('numpop'))
    biodata.variables['age'].long_name = 'fish age'
    biodata.variables['age'].units = 'yr'

    biodata.createVariable('age_bins', 'f8', ('time', 'maxage', 'nstocks'))
    biodata.variables['age_bins'].long_name = 'biomass in size bins'
    biodata.variables['age_bins'].units = 'yr'

    # Fish size of target harvest
    biodata.createVariable('wfish', 'f8', ('time'))
    biodata.variables['wfish'].long_name = 'fish size of target harvest'
    biodata.variables['wfish'].units = 'kg'

    # Resource
    biodata.createVariable('resource', 'f8', ('time', 'nstocks'))
    biodata.variables['resource'].long_name = 'resource'
    biodata.variables['resource'].units = 'units'

    # Mortality
    biodata.createVariable('mortality', 'f8', ('time', 'twos'))
    biodata.variables['mortality'].long_name = 'mortality biomass and number'
    biodata.variables['mortality'].units = 'kg and #'

    # Mortality by size
    biodata.createVariable('mort_bins', 'f8', ('time', 'numbins', 'nstocks'))
    biodata.variables['mort_bins'].long_name = 'mortality by size bins'
    biodata.variables['mort_bins'].units = '#'

    biodata.createVariable('mid_bins', 'f8', ('numbins'))
    biodata.variables['mid_bins'].long_name = 'middle of mortality bins'
    biodata.variables['mid_bins'].units = 'kg'

    return biodata


def compute_wtMat(asympLen, growthCoef, lenWtCoef, lenWtPower, maxage):
    yrage = np.linspace(1, maxage, 120)
    asympWt = (lenWtCoef * asympLen ** lenWtPower) / 1000

    length = asympLen * (1 - np.exp(-growthCoef * yrage))

    weight = (lenWtCoef * length ** lenWtPower) / 1000

    growth = asympWt * lenWtPower * growthCoef * (weight/asympWt) ** (1 - 1 / lenWtPower) * (1 - (weight / asympWt) ** (1 / lenWtPower))
    wtMat = weight[growth == np.max(growth)]
    agemat = yrage[growth == np.max(growth)]

    return wtMat
