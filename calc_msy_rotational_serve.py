#!/usr/bin/python
#Script to run multiple fisheries model

import os
from msy_stocks_rev14_connect import compute_pop_msy
import numpy as np
import pandas as pd
from datetime import datetime

#rev10 is most recent and used for data and figures

def calc_msy(outdir,fishdata,kk):

    minfiles = 1 #minimum files per simulation
    nstocks = 6 #number of stocks
    connectivity = 0 #0 for random, 1 for defined
    
    if connectivity>0:
        conn_data=pd.read_excel("/Users/bwoodson/Documents/UGA/COBIA_LAB/Projects/MSY_Sustainability/msyandpopdynamics/connectivity.xlsx",sheet_name='conn_matrix')
    
        conn_matrix=conn_data.to_numpy()
        conn_matrix=conn_matrix[:,1:]
        print(conn_matrix)
        
    else:
        conn_matrix=0
    
    nsamp = 1
    numit = 6 #number of total simulations to run
    nmin = 1
    
    #grab data from fishdata array
    species = fishdata['species'][kk]
    species = species.split()
    species = species[0] + '_' + species[1]
    
    xtest=0 #flag for simulation run
    oo=0 #current iteration number
    print ('%d ' % kk + species + ' started...')
     
    while xtest==0:
        
        rotational = 0
        
        Winf=-1 #set Winf to -1 to initialize
        W0=1 #set W0 to 1 to initialize
        
        
                #set growth parameters based on data
        #Linf = 1.*fishdata['Lmean'][kk] + 1.*np.random.randn()*fishdata['Lstd'][kk]/2 #asymptotic length
        Linf=np.zeros(2)
        Linf[0]=fishdata['Lmean'][kk]
        Linf[1]=fishdata['Lstd'][kk]
        #K=0
        #while K<=0:
        #K = fishdata['Kmean'][kk] + 0.*np.random.randn()*fishdata['Kstd'][kk]/4 #VBGF growth parameter
        K=np.zeros(2)
        K[0]=fishdata['Kmean'][kk]
        K[1]=fishdata['Kstd'][kk]

        b = fishdata['bmean'][kk] + 0.*np.random.randn()*fishdata['bstd'][kk]/20 #length-weight power relationship
        
        #calculate coefficient (a) of length-weight power relationship based on value of b
        sl=np.zeros([2])
        sl[0]=fishdata['slope'][kk]
        sl[1]=fishdata['inter'][kk]
        a = 10**(sl[0]*b+sl[1])
        
        maxage = fishdata['max age'][kk] #maximum age
        maxage=int(maxage) #convert max age to integer
        
        #set reproduction scaling (from earlier sensitivity experiments (see Fig S3)
        reprodmax = -1.*(np.log10(maxage/K[0]))-0.25
        reprodmin = -1.*(np.log10(maxage/K[0]))-4.75
    
        #set number of steps for evaluating recruitment rate and fishing rate
        rstep=48
        fstep=25
        
        #set recruitment values in log space
        reprodstp = np.logspace(reprodmin,reprodmax,rstep)
        
        #set biomass target for each level of fstep for surplus production curves
        btarget = np.linspace(1,0,fstep)
        
        #set asymptotic mass, initial mass at recruitment, and min size for capture
        Winf = (a*Linf[0]**b)/1000
        W0 = Winf*(1-np.exp(-K*0.25))**b
        minsize = 0.2*Winf #minimum size caught in kg
        
        #set background resource value to scale as Winf to constrain run time
        R = np.floor(400*Winf**1.2)

       #set file directories if needed
        if not (np.isnan(Linf[0]) or np.isnan(K[0]) or np.isnan(b) or np.isnan(maxage)):
            if not os.path.exists(outdir + species) and oo==0:
                os.makedirs(outdir + species)
                print('Creating data folder: ' + 'model_output/' + species)
            elif oo==0:
                filelist = [ f for f in os.listdir(outdir + species) if f.endswith('.nc') ]
                for f in filelist:
                    os.remove(os.path.join(outdir + species, f))
            
            rslap = 1 #recruitment estimate flag
             
            #set index for recruitment loop
            ll=0
            #set fishing rate to zero for recruitment loop
            f0=np.zeros([nstocks])
            
            #estimate minimum viable recruitment rate
            while rslap == 1:
                if ll<rstep:
                    reprodper=reprodstp[ll]
                    f0=np.zeros([nstocks])
                    rslap = compute_pop_msy(outdir,f0,nstocks,nstocks,species,Linf,K,a,b,maxage,minsize,reprodper,R,0,oo,1,1,0,0,0.0,0,conn_matrix,0)
                    minrec=1.*reprodper
                    ll=ll+1
                else:
                    minrec=1.*reprodstp[rstep-1]
                    rslap=0
            
            minrec=1.2*minrec
            
            #print('recruitment finished...rate = ',minrec)
            #initialize maximum fishing rate
            maxfish=0
            #set index for maximum fishing rate loop
            fslap = 1
            
            #estimate maximum fishing rate
            while fslap == 0:
                    f0=np.zeros([nstocks])+maxfish
                    fslap = compute_pop_msy(outdir,f0,nstocks,nstocks,species,Linf,K,a,b,maxage,minsize,minrec,R,0,oo,1,0,0,0,.5,0,conn_matrix,0)
                    maxfish = maxfish +.01
                    
            #maxfish = 2*maxfish
            #set index for main model run over various fishing rates
            stocktest = 0
            
            #set fishing rate vector based on maximum fishing rate and steps
            mfstp = np.linspace(0,maxfish,fstep)
            mfstp[0] = 0
            
            rotation=np.arange(0,fstep)
            rotation=np.append(rotation,500)
            
            #set number of stocks to be fished initially to one
            nfish=nstocks
            
            #print('max fishing rate finished...rate = ',maxfish)
            
            #set parameters
            rvar = 0.5
            msave=1
            btarget=0
            rptest=0
            environ=1
            ft=0
            
            #perform model simulations looping over number of fished stockes and fishing rate
            while stocktest == 0:
                    
                for ii in range(0,fstep+1): #41 for complete runs
                
                    #f0=np.zeros([nstocks])
                    f0[0:nfish] = 0.1#*maxfish
                    slap = compute_pop_msy(outdir,f0,nstocks,nfish,species,Linf,K,a,b,maxage,minsize,minrec,R,msave,oo,btarget,rptest,environ,ft,rvar,connectivity,conn_matrix,rotation[ii])

                stocklist = [ g for g in os.listdir(outdir + species) if g.endswith('%d' % nfish,19,20) and g.endswith('_' + '%d' % oo + '.nc')]
                
                stock_files = len(stocklist)

                #check to see if minimum files for each simulation is reached to estimate surplus prodction curves
                if stock_files>=minfiles:
                    nfish=nfish+1
                    if nfish>nstocks:
                        stocktest = 1
                else:
                    stocktest=1
            
            #check to see if number of files is sufficient based on minfiles and number of stocks, if reached proceed to next iteration, if not restart
            datalist = [ f for f in os.listdir(outdir + species) if f.endswith('_' + '%d' % oo + '.nc') ]
            number_files = len(datalist)
                
            if number_files<minfiles:
                for f in datalist:
                    os.remove(os.path.join(outdir + species, f))
            else:
                oo=oo+1
                if oo>=(numit):
                    xtest=1
                    
        else: #flag for data not sufficient and exit
            print('Insufficient Data...not running ' + species)
            xtest=1

    print ('%d ' % kk + species + ' is done.')
        


