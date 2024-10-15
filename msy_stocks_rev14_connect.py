#!/usr/bin/python
#Individual-based population dynamics model
# 20 Nov 21 - added fishing adjustment to insure B/K levels off

import datetime as dt
import numpy as np
import netCDF4 as nc

def compute_pop_msy(outdir,f0,nstocks,nfish,species,Linf,K,a,b,maxage,minsize,reprodper,R,msave,niter,btarget,rptest,environ,ft,rvar,connectivity,conn_matrix,rotation):
    #print (' ')
    #print ('MSY-IBM model started at ', dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    #outdir='/Volumes/COBIAT8/msy_sustain_paper/model_output2/'

    outfile=outdir + species + '/msy_stocks_' + '%d' % f0.size + '_nfish_' + '%d' % nfish + '_mfish_' + '%.4f' % np.max(f0) + '_rot_' + '%03d' % rotation + '_' + '%d' % niter + '.nc' #'_rec_' + '%.4f' % reprodper
    
    #initialize model
    #print ('Initializing model...')
    
    #initial conditions
    R0 = R/nstocks #resource availability
    if environ == 1:
        env=0.05 #enviromental scaling effect
        rho=0.25 #autocorrelation factor
    else:
        env=0
        rho=0
        
    #rotational closure parameters
    RM=0
    num_close=1
    
    N0 = int(400) #initial population size
    slap = 0 #restart flag
    #f0 = 0.08 #base fishing rate yr^-1
    delt = 1. #timestep in years
    nyr = 300 #length of simulation in years
    if rptest == 1:
        nyr=5*maxage
        if nyr < 200:
            nyr = 300
          
    tlength = int(np.ceil(nyr/delt))
    yy=np.zeros([tlength,1])
    
    #reprodper = .016 #percentage of reproduction that turns into new recruits (.03-.06)
    scf = 2
    szfc = 6
    nbins = 24
    #minsize = 4.
    
    #fish species growth parameters
    fish=np.zeros([tlength,N0])
    stock=np.random.randint(0,nstocks, size=N0)
    age=np.random.randint(1,maxage, size=N0)
    sex=np.random.randint(0,2,size=N0) #0 - male 1 - female
    age0=0.25
    maturity=np.zeros([N0])

    #gscale=.4
    gvar = K[1]*np.random.randn(N0)/2
    gvar[gvar<=-K[1]]=0
    Linfi=Linf[0]+Linf[1]*np.random.randn(N0)
    Linfi[Linfi<0]=Linf[0]
    
    Winf = (a*Linfi**b)/1000 #asymptotic weight based on Winf=aLinf^b
    Wmat = compute_wmat(Linf[0],K[0],a,b,maxage)
    W0 = Winf*(1-np.exp(-(K[0]+gvar)*age0))**b
    
    bin_min=0.1*(a*Linf[0]**b)/1000*(1-np.exp(-(K[0])*age0))**b
    bin_max=10*(a*Linf[0]**b)/1000
    
    fish[0,:]=Winf*(1-np.exp(-(K[0]+gvar)*age))**b
    
    m00=0.95
    
    m0 = m00*np.mean(W0)**(0.25) #scaled so that age 1 mortality is m00
    #m0 = 0.1*W0**(0.25)
    
    newgrowth=np.zeros([tlength])
    reprod_out=np.zeros([tlength])
    reproduction=np.zeros([tlength,2])
    preyconsumed=np.zeros([tlength])
    
    resource=np.zeros([tlength,nstocks])
    eps=np.zeros([tlength])

    Wfish=np.zeros([tlength])
    catch=np.zeros([tlength,nstocks,2])
    mortality=np.zeros([tlength,2])
    recruits=np.zeros([tlength])
    popsize=np.zeros([tlength,1])
    biomass=np.zeros([tlength,1])
    sex_ratio=np.zeros([tlength,1])
    stock_size=np.zeros([tlength,nstocks])
    stock_biomass=np.zeros([tlength,nstocks])
    stock_rec=np.zeros([tlength,nstocks])
    reprod_tot=np.zeros([tlength,nstocks])

    popsize[0]=N0
    biomass[0]=sum(fish[0,:])
    
    for kk in range(0,nstocks):
        stock_size[0,kk]=fish[0,stock==kk].size
        stock_biomass[0,kk]=np.sum(fish[0,stock==kk])
        stock_rec[0,kk]=0
    
    mort_bin=np.zeros([tlength,nbins,nstocks])
    pop_bin=np.zeros([tlength,nbins,nstocks])
    biom_bin=np.zeros([tlength,nbins,nstocks])
    age_bin=np.zeros([tlength,maxage,nstocks])
    catch_bin=np.zeros([tlength,nbins,nstocks])
    reprod_bin=np.zeros([tlength,nbins,nstocks])

    [nn,numfish]=fish.shape
    
    #print ('Starting model run...')
    for ii in range(1,nyr):
        if ft==0:
            if ii<100:
                ff=np.zeros([nstocks])
                mnpop=1
            #rotational closure
            elif ii>=100 and rotation > 0:
                ff=f0[:]*nstocks/(nstocks-1)
                if ii>0 and np.remainder(ii,rotation)==0:
                    RM=RM+1
                    if RM>(nstocks-1):
                        RM=0
                    ff[RM]=0
                else:
                    ff[RM]=0
            else:
                ff=f0
                mnpop=np.mean(popsize[ii-5:ii])/np.mean(popsize[40:50])
                #ff=np.zeros([nstocks]) #if examining fishery recovery

        if mnpop<btarget:
            ff=btarget*ff
            
        netgrowth=np.zeros([numfish])
        consumed=np.zeros([numfish])
        reprod=np.zeros([numfish])
        eps[ii]=rho*eps[ii-1]+np.sqrt(1-rho**2)*env*R0*np.random.normal();
        resource[ii,:]=R0+eps[ii]
        
        dead=np.zeros([numfish])
        fished=np.zeros([numfish])
        
        Wfish[ii]=2*np.mean(fish[ii-1,:])
        
        order=np.random.permutation(numfish)
        
        #if rptest==0 and ii==230:
        #    idd=np.random.permutation(np.arange(numfish))[:int(np.random.uniform(0,1)*numfish)]
        #    dead[idd]=1
        
        for kk in range(0,numfish):
        
            jj=order[kk]
            #jj=kk
            
            #if not np.isfinite(fish[ii-1,jj]) or fish[ii-1,jj]<=0:
            #    print(fish[ii-1,jj])
            if dead[jj]==0:
            
                bio_order=np.random.permutation(3)
                
                for mm in bio_order:
                    if mm==0:
                        #individual consumption of resource
                        consumption=6.0*fish[ii-1,jj]**1.00
                        production=3.0*fish[ii-1,jj]**0.75
                        metabolism=consumption-production
                        
                        if resource[ii,stock[jj]]>=consumption:
                            avail_energy=consumption
                        elif fish[ii-1,jj]>100*np.mean(W0):
                            avail_energy=resource[ii,stock[jj]]
                            cann=np.abs((fish[ii-1,:]>0)*(fish[ii-1,:]<=.01*fish[ii-1,jj])*(1-dead))
                            idcan=np.asarray(cann.nonzero())
                            [dd,ee]=idcan.shape
                            for cc in range(0,ee):
                                eatid=idcan[0,cc]
                                cannibal=np.random.random()
                                cann_rate=1.0*fish[ii-1,eatid]**(-0.25)
                                cann_prob=1-np.exp(-cann_rate*delt)
                                if cannibal<=cann_prob and avail_energy<consumption:
                                    avail_energy=avail_energy+fish[ii-1,eatid]
                                    dead[eatid]==1
                        else:
                            consumption=resource[ii,stock[jj]]
                            avail_energy=resource[ii,stock[jj]]
                        
                        consumed[jj]=consumption

                       #growth and reproduction
                        if (fish[ii,jj]<Wmat and avail_energy<metabolism):
                            mortality[ii,0]=mortality[ii,0]+1
                            mortality[ii,1]=mortality[ii,1]+fish[ii-1,jj]
                            dead[jj]=1
                            consumption=0
                        elif avail_energy<metabolism :
                            growth=0
                            reprod[jj]=0
                            netgrowth[jj]=avail_energy-metabolism
                        else:
                            pconsum=(avail_energy-metabolism)/production
                            growth=pconsum*Winf[jj]*b*(K[0]+gvar[jj])*(fish[ii-1,jj]/Winf[jj])**(1-1/b)*(1-(fish[ii-1,jj]/Winf[jj])**(1/b))
                            netgrowth[jj]=growth*delt
                            if maturity[jj]==1 and sex[jj]==1:
                                massegg=(.008*fish[ii-1,jj]**.109)/1000**2
                                #massegg=(0.045*fish[ii-1,jj]**.11+(1.025-(.045*fish[ii-1,jj]**.14+.006))*(0.395*fish[ii-1,jj]**.14))/1000**2
                                reprod[jj]=int((avail_energy-growth)/massegg)
                                if reprod[jj]<0:
                                    reprod[jj]=0
                                        
                        fish[ii,jj]=fish[ii-1,jj]+netgrowth[jj]
                    
                    if mm==1:
                        #naturalmortality (including predation
                        #morprob=m0*fish[ii,jj]**(-0.25)+1./(1+np.exp(-0.6*(age[jj]-0.9*maxage)))
                     
                        mort=m0*fish[ii-1,jj]**(-0.25)+1./(1+np.exp(-0.6*(age[jj]-maxage)))
                        
                        morprob=1-np.exp(-mort*delt)
                        
                        death=np.random.random()
                        
                        if death<morprob:
                            mortality[ii,0]=mortality[ii,0]+1
                            mortality[ii,1]=mortality[ii,1]+fish[ii-1,jj]
                            dead[jj]=1
                            consumption=0
                        
                    if mm==2:
                        #fishing mortality
                        caught=np.random.random()

                        if ff[stock[jj]]>0:
                            fishmort=scf*ff[stock[jj]]/(1+np.exp(-0.05*(fish[ii-1,jj]-Wfish[ii])))
                            fishmprob=1-np.exp(-fishmort*delt)
                        else:
                            fishmort=0
                            fishmprob=0

                        if caught<fishmprob and fish[ii,jj]>minsize:
                            fished[jj]=1
                            dead[jj]=1
                            catch[ii,stock[jj],0]=catch[ii,stock[jj],0]+1
                            catch[ii,stock[jj],1]=catch[ii,stock[jj],1]+fish[ii,jj]
                            consumption=0
                        
                age[jj]=age[jj]+1
                
                #check if fish reaches maturity
                if maturity[jj]==0:
                    #print(np.exp(-2.*(fish[ii,jj]-Wmat)))
                    pmat=1./(1+np.exp(-2.*(fish[ii,jj]-Wmat)))
                    ptest=np.random.randn()
                    if ptest>pmat:
                        maturity[jj]=1

                resource[ii,stock[jj]]=resource[ii,stock[jj]]-consumption
                if resource[ii,stock[jj]]<0:
                    resource[ii,stock[jj]]=0
                
        #update mass and age data for dead, caught, or new individuals
        idx=np.asarray(dead).nonzero()
        tempfish=fish[ii,idx]
        nsx=stock[idx]

        #bin_ranges=np.concatenate((np.logspace(np.log10(bin_min),0,int(0.3*nbins)),np.linspace(1,bin_max,int(0.7*nbins+1)+1)))
        
        #print(bin_ranges)
        #bin_ranges=np.linspace(bin_min,bin_max,nbins+1)
        
        bin_ranges=np.logspace(np.log10(bin_min),np.log10(bin_max),nbins+1)
        
        for kk in range(0,nstocks):
            if nsx[nsx==kk].size>0:
                #[mort_bin[ii,:,kk], bin_edges]=np.histogram(tempfish[0,nsx==kk],bins=nbins,range=[0, np.mean(Winf*1.1)]) #linear
                [mort_bin[ii,:,kk], bin_edges]=np.histogram(tempfish[0,nsx==kk],bins=bin_ranges) #log
        
        idf=np.asarray(fished).nonzero()
        nst=stock[idf]
        tempfish=fish[ii,idf]
        
        fish = np.delete(fish, idx, axis=1)
        age = np.delete(age, idx)
        sex = np.delete(sex, idx)
        stock = np.delete(stock, idx)
        reprod = np.delete(reprod,idx)
        maturity = np.delete(maturity,idx)
        Winf = np.delete(Winf,idx)
        W0 = np.delete(W0,idx)
        gvar = np.delete(gvar,idx)

        for kk in range(0,nstocks):
            if nst[nst==kk].size>0:
                #[catch_bin[ii,:,kk], bin_edges]=np.histogram(tempfish[0,nst==kk],bins=nbins,range=[0, np.mean(Winf*1.1)]) #linear
                [catch_bin[ii,:,kk], bin_edges]=np.histogram(tempfish[0,nst==kk],bins=bin_ranges) #log

        for kk in range(0,nstocks):
            for ll in range(0,nbins):
                ids=np.where(np.logical_and(np.logical_and(fish[ii-1,:]>=bin_edges[ll], fish[ii-1,:]<bin_edges[ll+1]),stock==kk))
                if ids[0].shape[0]>0:
                    biom_bin[ii,ll,kk]=np.sum(fish[ii-1,ids])
                    reprod_bin[ii,ll,kk]=np.sum(reprod[ids])
                    mid_bins=bin_edges[0:-1]+(bin_edges[1]-bin_edges[0])/2.

        for kk in range(0,nstocks):
            #[pop_bin[ii,:,kk], dumb]=np.histogram(fish[ii,stock==kk],bins=nbins,range=[0, np.mean(Winf*1.1)]) #linear
            [pop_bin[ii,:,kk], dumb]=np.histogram(fish[ii,stock==kk],bins=bin_ranges) #log
            [age_bin[ii,:,kk], dumb]=np.histogram(age[stock==kk],bins=maxage,range=[0.5, maxage+0.5])

        [nn,numfish]=fish.shape
        if (numfish==0):
            slap=1
            #return slap
            break
            
        #update mass and age data for new recruits
        rvar=0.5
        #print(rvar,sigrec)
        rep_scale=1
        
        if connectivity>0:
            reprod_matrix=np.zeros([nstocks,nstocks])
            numrec_stock=np.zeros([nstocks,1])
            
            for kk in range(0,nstocks):
                reprod_tot[ii,kk]=np.sum(reprod[stock==kk])
                for mm in range(0,nstocks):
                    reprod_matrix[kk,mm]=conn_matrix[kk,mm]*reprod_tot[ii,kk]
                    #print(kk,mm,reprod_matrix[kk,mm])
                sigrec=np.random.lognormal(0.,rvar)
                numrec_stock[kk]=int(sigrec*np.sum(reprodper/1e5*reprod_matrix[:,kk]))
                #print(kk,numrec_stock[kk])
                
            numrec=int(np.sum(numrec_stock))
            #print(numrec)
        else:
            sigrec=np.random.lognormal(0.,rvar)
            numrec=int(sigrec*reprodper/1e5*np.sum(reprod))

    
        #rep_scale=1/(1+np.exp(.01*(np.sum(fish[ii,:])-R0/12))) #density dependence in plankton
        #rep_scale=1*(400-np.sum(fish[ii,:]))/(400)
        #rep_scale=99*np.exp(-.01*np.sum(fish[ii,:]))+1
    
        #numrec=int(sigrec*(rep_scale*reprodper/1e5)*np.sum(reprod))
        #for mm in range(0,nstocks)
        #    numrec[mm]=int(reprodper/1e5*np.sum(reprod[stock==mm]))
        
        #numrec=int(0.3*popsize[ii-1]) #allometric
        #numrec=int(0.25*(popsize[ii-1]/(1+0.0005*popsize[ii-1]))) #beverton-holt
        #numrec=int(0.25*popsize[ii-1]*np.exp(-0.001*popsize[ii-1])) #ricker

        if numrec>0:
        
            if connectivity>0:
                prob_stock=np.cumsum(numrec_stock/numrec)
                #print(prob_stock)
                stockn=np.random.random(size=numrec)
                newstock=np.zeros([numrec],dtype=int)
                
                #print(prob_stock)

                for rr in range(0,nstocks):
                    idr=np.asarray(np.nonzero(stockn<=prob_stock[rr]))
                    stockn[idr]=2
                    newstock[idr]=int(rr)
            else:
                newstock=np.random.randint(0,nstocks, size=numrec)
                
            
            reproduction[ii,0]=numrec
            newrec=np.zeros([nn,numrec])
            newgvar=K[1]*np.random.randn(numrec)
            newgvar[newgvar<=-K[0]]=0
            newage=np.zeros([numrec])
            newsex=np.random.randint(0,2,size=numrec)
            newLinf=Linf[0]+Linf[1]*np.random.randn(numrec)
            newLinf[newLinf<0]=Linf[0]
            newWinf = (a*newLinf**b)/1000
            newW0 = newWinf*(1-np.exp(-(K[0]+newgvar)*age0))**b
            newrec[ii,:] = newW0
            reproduction[ii,1]=np.sum(newrec[ii,:])
            
            fish = np.append(fish,newrec,axis=1)
            age = np.append(age,newage)
            sex = np.append(sex,newsex)
            stock = np.append(stock,newstock)
            maturity = np.append(maturity,np.zeros([numrec]))
            Winf = np.append(Winf,newWinf)
            W0 = np.append(W0,newW0)
            gvar = np.append(gvar,newgvar)
                
        [nn,numfish]=fish.shape

        #calculate population parameters for data storage
        preyconsumed[ii]=np.sum(consumed)
        newgrowth[ii]=np.sum(netgrowth)
        reprod_out[ii]=np.sum(reprod)
        yy[ii]=ii
        if numrec>0:
            popsize[ii]=numfish-numrec
            biomass[ii]=np.sum(fish[ii,:])-np.sum(newrec[ii,:])
        else:
            popsize[ii]=numfish
            biomass[ii]=np.sum(fish[ii,:])
        sex_ratio[ii]=np.sum(sex)/sex.shape
        
        for kk in range(0,nstocks):
            stock_size[ii,kk]=fish[ii,stock==kk].size
            stock_biomass[ii,kk]=np.sum(fish[ii,stock==kk])
            if numrec>0:
                stock_rec[ii,kk]=newrec[ii,newstock==kk].size
                
        #report model progress
        #if np.mod(ii+1,np.floor(nyr*.1))==0:
            #print ('Run is',(float((ii+1))/tlength*1.0)*100.0,'% done and population is',numfish,' with a biomass of ', int(np.ceil(biomass[ii])))

    #create output file
    if msave>0:
        biodata=create_popdynnc(outfile,nyr,numfish,nbins,maxage,nstocks)

        #write model output to
        biodata.variables['ftime'][:]=yy
        biodata.variables['fishing_rate'][:]=f0
        biodata.variables['reprod_rate'][:]=reprodper
        biodata.variables['fish'][:,:]=fish
        biodata.variables['popsize'][:]=popsize
        biodata.variables['biomass'][:]=biomass
        biodata.variables['consumption'][:]=preyconsumed
        biodata.variables['newgrowth'][:]=newgrowth
        biodata.variables['reprod_out'][:]=reprod_out
        biodata.variables['reproduction'][:,:]=reproduction
        biodata.variables['catch'][:,:,:]=catch
        biodata.variables['age'][:]=age
        biodata.variables['wfish'][:]=Wfish
        biodata.variables['resource'][:,:]=resource
        biodata.variables['mortality'][:,:]=mortality
        biodata.variables['stock_size'][:,:]=stock_size
        biodata.variables['stock_biomass'][:,:]=stock_biomass
        biodata.variables['stock_recruit'][:,:]=stock_rec
        biodata.variables['sex_ratio'][:]=sex_ratio

        biodata.variables['mort_bins'][:,:,:]=mort_bin
        biodata.variables['catch_bins'][:,:,:]=catch_bin
        biodata.variables['pop_bins'][:,:,:]=pop_bin
        biodata.variables['biomass_bins'][:,:,:]=biom_bin
        biodata.variables['reprod_bins'][:,:,:]=reprod_bin
        biodata.variables['age_bins'][:,:,:]=age_bin
        biodata.variables['mid_bins'][:]=mid_bins
        biodata.variables['rotation_rate'][:]=rotation
        
        biodata.close()
        
    return slap

#Subroutine to create output netcdf file
def create_popdynnc(outfile,tlength,numpop,nbins,maxage,nstocks):
    biodata = nc.Dataset(outfile, 'w')
    biodata.createDimension('time', tlength)
    biodata.createDimension('numpop', numpop)
    biodata.createDimension('numbins', nbins)
    biodata.createDimension('maxage', maxage)
    biodata.createDimension('nstocks', nstocks)
    biodata.createDimension('ones', 1)
    biodata.createDimension('twos', 2)
    
    #Time
    biodata.createVariable('ftime', 'f8', ('time'))
    biodata.variables['ftime'].long_name = 'time since model start'
    biodata.variables['ftime'].units = 'years'

    #Fish Rate
    biodata.createVariable('fishing_rate', 'f8', ('nstocks'))
    biodata.variables['fishing_rate'].long_name = 'fishing rate in each stock'
    biodata.variables['fishing_rate'].units = '%'

    #Reproduction Rate
    biodata.createVariable('reprod_rate', 'f8', ('nstocks'))
    biodata.variables['reprod_rate'].long_name = 'mean survivial rate of larvae'
    biodata.variables['reprod_rate'].units = '%'
    
    #Rotation Rate
    biodata.createVariable('rotation_rate', 'f8', ('ones'))
    biodata.variables['rotation_rate'].long_name = 'period of time before reserve rotation'
    biodata.variables['rotation_rate'].units = 'years'
    
    #Fish Size
    biodata.createVariable('fish', 'f8', ('time','numpop'))
    biodata.variables['fish'].long_name = 'individual size through time'
    biodata.variables['fish'].units = 'kg'
    
    #Population Size
    biodata.createVariable('popsize', 'f8', ('time'))
    biodata.variables['popsize'].long_name = 'population size'
    biodata.variables['popsize'].units = '# of individuals'
    
    biodata.createVariable('pop_bins', 'f8', ('time','numbins','nstocks'))
    biodata.variables['pop_bins'].long_name = 'populations size in size bins'
    biodata.variables['pop_bins'].units = '#'
    
    #Sex Ratio
    biodata.createVariable('sex_ratio', 'f8', ('time'))
    biodata.variables['sex_ratio'].long_name = 'proportion female'
    biodata.variables['sex_ratio'].units = 'proportion'
    #Stock Size
    biodata.createVariable('stock_size', 'f8', ('time','nstocks'))
    biodata.variables['stock_size'].long_name = 'stock size'
    biodata.variables['stock_size'].units = '# of individuals'
    
    biodata.createVariable('stock_biomass', 'f8', ('time','nstocks'))
    biodata.variables['stock_biomass'].long_name = 'stock biomass'
    biodata.variables['stock_biomass'].units = '# of individuals'
    
    biodata.createVariable('stock_recruit', 'f8', ('time','nstocks'))
    biodata.variables['stock_recruit'].long_name = 'stock recruits'
    biodata.variables['stock_recruit'].units = '# of individuals'
    
    #Biomass
    biodata.createVariable('biomass', 'f8', ('time'))
    biodata.variables['biomass'].long_name = 'biomass'
    biodata.variables['biomass'].units = 'kg'
    
    biodata.createVariable('biomass_bins', 'f8', ('time','numbins','nstocks'))
    biodata.variables['biomass_bins'].long_name = 'biomass in size bins'
    biodata.variables['biomass_bins'].units = '#'
    
    #New growth
    biodata.createVariable('newgrowth', 'f8', ('time'))
    biodata.variables['newgrowth'].long_name = 'new biomass'
    biodata.variables['newgrowth'].units = 'kg'
    
    #Reproductive Output
    biodata.createVariable('reprod_out', 'f8', ('time'))
    biodata.variables['reprod_out'].long_name = 'reproductive output'
    biodata.variables['reprod_out'].units = 'number of eggs'
    
    #Consumption
    biodata.createVariable('consumption', 'f8', ('time'))
    biodata.variables['consumption'].long_name = 'new biomass'
    biodata.variables['consumption'].units = 'kg'
    
    #Reproduction
    biodata.createVariable('reproduction', 'f8', ('time','twos'))
    biodata.variables['reproduction'].long_name = 'biomass and number'
    biodata.variables['reproduction'].units = 'kg and #'
    
    biodata.createVariable('reprod_bins', 'f8', ('time','numbins','nstocks'))
    biodata.variables['reprod_bins'].long_name = 'reproduction in size bins'
    biodata.variables['reprod_bins'].units = '#'
    
    #Catch
    biodata.createVariable('catch', 'f8', ('time','nstocks','twos'))
    biodata.variables['catch'].long_name = 'catch biomass and numbers'
    biodata.variables['catch'].units = 'kg and #'
    
    biodata.createVariable('catch_bins', 'f8', ('time','numbins','nstocks'))
    biodata.variables['catch_bins'].long_name = 'catch in size bins'
    biodata.variables['catch_bins'].units = '#'
    
    #Age
    biodata.createVariable('age', 'f8', ('numpop'))
    biodata.variables['age'].long_name = 'fish age'
    biodata.variables['age'].units = 'yr'
    
    biodata.createVariable('age_bins', 'f8', ('time','maxage','nstocks'))
    biodata.variables['age_bins'].long_name = 'biomass in size bins'
    biodata.variables['age_bins'].units = 'yr'
    
    #Fish size of target harvest
    biodata.createVariable('wfish', 'f8', ('time'))
    biodata.variables['wfish'].long_name = 'fish size of target harvest'
    biodata.variables['wfish'].units = 'kg'
    
    #Resource
    biodata.createVariable('resource', 'f8', ('time','nstocks'))
    biodata.variables['resource'].long_name = 'resource'
    biodata.variables['resource'].units = 'units'
    
    #Mortality
    biodata.createVariable('mortality', 'f8', ('time','twos'))
    biodata.variables['mortality'].long_name = 'mortality biomass and number'
    biodata.variables['mortality'].units = 'kg and #'
    
    #Mortality by size
    biodata.createVariable('mort_bins', 'f8', ('time','numbins','nstocks'))
    biodata.variables['mort_bins'].long_name = 'mortality by size bins'
    biodata.variables['mort_bins'].units = '#'
    
    biodata.createVariable('mid_bins', 'f8', ('numbins'))
    biodata.variables['mid_bins'].long_name = 'middle of mortality bins'
    biodata.variables['mid_bins'].units = 'kg'
    
    return biodata

def compute_wmat(Linf,K,a,b,maxage):

    yrage = np.linspace(1,maxage,120)
    Winf = (a*Linf**b)/1000

    L=Linf*(1-np.exp(-K*yrage))

    W = (a*L**b)/1000

    growth=Winf*b*K*(W/Winf)**(1-1/b)*(1-(W/Winf)**(1/b))
    Wmat = W[growth==np.max(growth)]
    agemat = yrage[growth==np.max(growth)]

    return Wmat






