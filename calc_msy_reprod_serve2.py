#!/usr/bin/python
#Script to run multiple fisheries model

import os
from msy_stocks_rev10_18Dec20 import compute_pop_msy
import numpy as np
import pandas as pd
from datetime import datetime


#rev7 is currently used for data and figures
#rev9 is most recent
#msy_stocks_rev1 is for multiple stocks

def calc_msy(outdir,fishdata,kk):

    #outdir='/Volumes/COBIAT8/msy_sustain_paper/model_output_rev9/'
    #outdir='/Users/bwoodson/Documents/UGA/COBIA_LAB/Projects/MSY_sustainability/msyandpopdynamics/rev9test.nosync/'

    #fishdata=pd.read_excel('/Users/bwoodson/Documents/UGA/COBIA_LAB/Projects/MSY_Sustainability/msyandpopdynamics/fish_growth_data.xlsx',sheet_name='fish_growth_data')
    #print(numspec)
    minfiles = 12
    nstocks = 4
    nsamp = 1
    numit = 32

    species = fishdata['species'][kk]
    species = species.split()
    species = species[0] + '_' + species[1]

    xtest=0
    oo=0
    print ('%d ' % kk + species + ' started...')
     
    while xtest==0:
    
        Winf=-1
        W0=1
        
        Linf = 1.*fishdata['Lmean'][kk] + 0.*np.random.randn()*fishdata['Lstd'][kk]/2
        
        #K=0
        #while K<=0:
        K = fishdata['Kmean'][kk] + 0.*np.random.randn()*fishdata['Kstd'][kk]/4
        
        b = fishdata['bmean'][kk] + 0.*np.random.randn()*fishdata['bstd'][kk]/20
        
        sl=np.zeros([2])
        sl[0]=fishdata['slope'][kk]
        sl[1]=fishdata['inter'][kk]
        a = 10**(sl[0]*b+sl[1])
        maxage = fishdata['max age'][kk]
        
        reprodmax = -1.*(np.log10(maxage/K))-0.25
        reprodmin = -1.*(np.log10(maxage/K))-4.75
        #reprodmin=-10
        #reprodmax=0
        rstep=48
        fstep=1
        reprodstp = np.logspace(reprodmin,reprodmax,rstep)
        btarget = np.linspace(1,0,fstep)
        
        SLP = 0.29
        INT = -0.25
        
        Winf = (a*Linf**b)/1000
        W0 = Winf*(1-np.exp(-K*0.25))**b
        minsize = 0.2*Winf #minimum size caught in kg
        
        R = np.floor(1000*Winf**1.2)
        #print(Winf,R)
        if not (np.isnan(Linf) or np.isnan(K) or np.isnan(b) or np.isnan(maxage)):
            if not os.path.exists(outdir + species) and oo==0:
                os.makedirs(outdir + species)
                print('Creating data folder: ' + 'model_output/' + species)
            elif oo==0:
                filelist = [ f for f in os.listdir(outdir + species) if f.endswith('.nc') ]
                for f in filelist:
                    os.remove(os.path.join(outdir + species, f))
            
            rslap = 1
             
            ll=0
            f0=np.zeros([nstocks])
            
            maxage=np.int(maxage)
            
            #t1=now = datetime.now()
            while rslap == 1:
                if ll<rstep:
                    reprodper=reprodstp[ll]
                    f0=np.zeros([nstocks])
                    rslap = compute_pop_msy(outdir,f0,nstocks,nstocks,species,Linf,K,a,b,maxage,minsize,reprodper,R,0,oo,1,1,0)
                    minrec=1.*reprodper
                    ll=ll+1
                else:
                    minrec=1.*reprodstp[rstep-1]
                    rslap=0
            
            #rfile=open('reptest2.txt','a')
            #rfile.write('%f,' % Winf + '%f,' % K + '%d,' % maxage + '%0.9f\n' % minrec)
                    
            #t2=datetime.now()
            #print(t2-t1,minrec)
            
            maxfish=0
            fslap = 0
            
            #t1=now = datetime.now()
            while fslap == 0:
                    f0=np.zeros([nstocks])+maxfish
                    fslap = compute_pop_msy(outdir,f0,nstocks,nstocks,species,Linf,K,a,b,maxage,minsize,minrec,R,0,oo,1,0,1)
                    maxfish = maxfish +.01
                    
            #t2=datetime.now()
            #print(t2-t1,maxfish)
            
            maxfish = 1.*maxfish
            
            stocktest = 0
            
            mfstp = np.linspace(0,maxfish,fstep)
            mfstp[0] = 0
            nfish=1
            
            while stocktest == 0:
                    
                for ii in range(0,fstep): #41 for complete runs
                
                    f0=np.zeros([nstocks])
                    f0[0:nfish] = (nstocks/nfish)*mfstp[ii]
                    slap = compute_pop_msy(outdir,f0,nstocks,nfish,species,Linf,K,a,b,maxage,minsize,minrec,R,1,oo,btarget[ii],0,0)

                stocklist = [ g for g in os.listdir(outdir + species) if g.endswith('%d' % nfish,19,20) and g.endswith('_' + '%d' % oo + '.nc')]
                
                stock_files = len(stocklist)
                    
                if stock_files>=minfiles:
                    nfish=nfish+1
                    if nfish>nstocks:
                        stocktest = 1
                else:
                    #stocklist = [ g for g in os.listdir(outdir + species) if g.endswith('_' + '%d' % oo + '.nc')]
                    #for g in stocklist:
                    #    os.remove(os.path.join(outdir + species, g))
                    stocktest=1
                    
            datalist = [ f for f in os.listdir(outdir + species) if f.endswith('_' + '%d' % oo + '.nc') ]
            number_files = len(datalist)
                
            if number_files<minfiles:
                for f in datalist:
                    os.remove(os.path.join(outdir + species, f))
            else:
                oo=oo+1
                if oo>=(numit):
                    xtest=1
                    
        else:
            print('Insufficient Data...not running ' + species)
            xtest=1

    print ('%d ' % kk + species + ' is done.')
        


