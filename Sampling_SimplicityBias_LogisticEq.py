# Author - Kamaludin Dingle

# This code is for analysing simplicity bias in the logistic equation. x_{n+1} = mu*x_n*(1-x_n), which is known to produce chatoic behaviour for ceratin starting x in [0,1) values and mu in [0,4]. Here we randomly choose starting x values and mu values, and for each we run the iterative map 30-50 times, and discretise into a binary string via the standard protocol (e.g. Ali Kanso & Nejib Smaoui 2007) via a threshold of 0.5.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import sys
PATH = '/Users/Kamal/Documents/Research/Computation/Python Stuff/'
sys.path.insert(0, PATH)

#import Complexity
import KC
#print (KC.calc_KC('010101110001')) # Test

#FIG_PATH = '/Users/Kamal/Documents/Research/Computation/Python Stuff/LogisticMapChaos/Figs/'
FIG_PATH  = '/Users/kamaludindingle/Documents/Research/Writing/LogisticEquation/Figs/'

# Notes:
# Looking at the initial segment, vs throwing away the first 100 iterates produces the same plots

#xstep = 0.99;
walklength = 10
print ('Output string length is',walklength)

N = int( 5*10**6 )
print ('Number of samples is ',N)

H_vals = [] # entropy of distn
m_vals = [] # slope
t_vals = [0.0,3.0, 3.57, 4.0]#[0.0,0.5, 1.0, 1.5,2.0, 2.5,3.0, 3.5, 3.6,3.7,3.8,3.85,3.9, 3.95,4.0]
t_vals_use = []# in case any t-values do not produce a slope (ie nan slope)
for t in t_vals:
    print ('\nt = ',t)

    print ('Sampling underway...')
    UpDnPatterns = []
    for samps in np.arange(N):
        #x0 = np.random.choice(np.arange(0,1+xstep,xstep))
        x0 = np.random.rand()
        #mu = 4*np.random.choice(np.arange(0,1+xstep,xstep))
        #t = 0.0#3.5, 3.75, 3.875, 4.0
        mu = t + (4-t)*np.random.rand()
        #mu = 4
        
        x = x0
        X =[]
        for j in np.arange(walklength):
            xnext = mu*x*(1-x) # if mu is 4, then chaotic on whole of [0,1], page 42 of Chaos and Chance
            #xnext = (2*x % 1)
            X.append(xnext)
            x = xnext
        
        #Y = [np.sign(X[t+1] - X[t]) for t in range(len(X)-1)]
        #Y = Y[100:140] # to test if simp bias only in initial segment
        # use >0.5 criterion instead
        Y = 1.0*(np.array(X)>=0.5)
        #UpDnStr = ''.join([(str(int(s))) for s in (np.sign(np.asarray(Y)+0.5)+1)/2])
        UpDnStr = ''.join([(str(int(s))) for s in Y])
        UpDnPatterns.append(UpDnStr)
        if (samps % 50000)==0 and samps>0:
            print (round(100.0*samps/N), ' % completed')



    N_UniqOuts = len(set(UpDnPatterns)) # number of unique outputs
    print ('Number of unique outputs is',N_UniqOuts)

    print ('Computing frequencies and complexities...')
    Freqs = []
    K = []
    Patterns = []
    for j,uni in enumerate(set(UpDnPatterns)):
        var1 = UpDnPatterns.count(uni)
        Freqs.append(var1)
        #K.append(Complexity.CLZ(uni))
        K.append(KC.calc_KC(uni))
        Patterns.append(uni)
        if (j % 1000)==0:
            print (round(100.0*j/N_UniqOuts), ' % completed')


    K_scaled = walklength * ( K-min(K) ) / ( max(K)-min(K) )

    K_scaled = np.round(K_scaled,2)

    P = 1.0*np.asarray(Freqs)/np.sum(Freqs)

    plt.figure()
    plt.semilogy(K_scaled,P,'o',color='green',ms=5, markeredgewidth=1.5)
    plt.xlabel(r'$\tilde{K}(x)$',fontsize=18)
    plt.ylabel(r'$P(x)$',fontsize=18)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.ylim([0.5*1/N,1])
    title_str='Samples= %d Length = %d t=%.2f' % (int(N), int(walklength),t)
    plt.title(title_str)
    #plt.show()

    # Prediction
    #N_O = len(K)

    #maxK = max(K)
    #a = (np.log2(N_O) + np.log2(max(P))) / (maxK-min(K))# magnitude of gradient
    #a = np.log2(N_O)/maxK
    #b = -np.abs(a)*min(K) - np.log2(max(P))
    #b = 0

    #K = np.asarray(K)
    #Pred = 2**(-a*K -b)
    Pred  = 2**-K_scaled
    plt.semilogy(K_scaled,Pred,label='Prediction',color='black',alpha=0.5)
    #plt.legend()
    #v = [0, 45, 10**(-1), 1]
    #plt.axis(v)
    plt.tight_layout()
    plt.show()
    plt.savefig(FIG_PATH + 'Logistic_'+str(walklength)+'_N'+str(N)+'_t'+str(t)+'.png')


    # save the data
    #OutFileName = 'GeneratedData/out_Logistic_Samps%d_len%d.txt' % (N,walklength)
    #np.savetxt(OutFileName,np.c_[K,Freqs])

    # Save outputs phenotypes also (For Chico)
    OutFileName = 'OutputData/out_Logistic_Phenos_Samps%d_len%d_t%.2f.txt' % (N,walklength,t)
    np.savetxt(OutFileName,np.c_[Patterns,K_scaled,Freqs],fmt=['%s', '%s', '%s'])

    
    # Find the actual slope of the upper bound. Only count if at least 10 outputs of a given complexity
    k_vals = []
    max_P_k_vals = []
    for k in np.unique(K_scaled):
        inds = np.where(K_scaled==k)[0]# find outputs with that complexity
        max_P_k = np.max(P[inds])
        if max_P_k >=10 * 1/N: # the output with max prob for that k appeared at least 10 times, to reduce statistical fluctuations
            k_vals.append(k)
            max_P_k_vals.append(max_P_k)

    # entropy of distn P
    H = -np.sum(P*np.log2(P))

    # slope m
    if len(k_vals)>=3: # to get some rasonable slope we need at least 3 points
        m,b=np.polyfit(k_vals,np.log10(max_P_k_vals),1)
        m_vals.append(m)
        H_vals.append(H)# store for each t value
        t_vals_use.append(t)
    else:
        m = np.nan




'''
plt.figure()
plt.scatter(H_vals,m_vals,c = t_vals_use)
plt.xlabel('Entropy (bits)',fontsize=18)
plt.ylabel('Upper bound slope',fontsize=18)
cbar = plt.colorbar()
cbar.set_label('t', rotation=0,fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig(FIG_PATH + 'Logistic_H_vs_slope_'+str(walklength)+'_N'+str(N)+'_t'+str(t)+'.png')

r = stats.pearsonr(H_vals,m_vals)
print('Correlation of entropy and slope is r=',np.round(r[0],2), ' p-val=',np.round(r[1],5))
'''
print ('Done.')
