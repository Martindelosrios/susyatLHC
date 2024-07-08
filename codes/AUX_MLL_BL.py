import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib import patches

import os
import math
import random

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, auc

import scipy.stats as st
from scipy.stats import poisson
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.optimize import fsolve

from xgboost import XGBRegressor, XGBClassifier



#####################
# BINARY CLASSIFIER #
#####################

def XG(X_train, X_test, X_val, Y_train, Y_test, Y_val, n_estimators=500,
        learning_rate=0.1,
        reg_lambda=0.0,reg_alpha=0.0,
        gamma=0.0,objective='binary:logistic',
        max_depth=5,
        early_stopping_rounds=50):
        
    # XGBOOST BINARY CLASSIFIER
    # Inputs:
		# X_train, X_test, X_val -> trainning, test, and validation datasets, shape: ( # event, # features )
		# Y_train, Y_test, Y_val -> trainning, test, and validation lavels,   shape: ( # event, 1 )
		# XGBoost parameters (see XGBoost documentation)
		
	# Outputs:
		# roc_auc_xg -> ROC_AUC value
		# y_pred_xg -> classifier output of the test dataset, shape: ( # events en X_test, )
    
    # CLASSIFIER
    classifier = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,reg_alpha=reg_alpha,
        gamma=gamma,objective='binary:logistic',
        max_depth=max_depth)

    # FIT
    classifier.fit(X_train,Y_train,eval_set=[(X_train, Y_train), (X_val, Y_val)],
            eval_metric='logloss',early_stopping_rounds=early_stopping_rounds,#early stopping
            verbose=True)

    # PREDICTIONS
    y_pred_xg = classifier.predict_proba(X_test).T[1]


    # ROC
    fpr_xg, tpr_xg, _ = roc_curve(Y_test, y_pred_xg)
    roc_auc_xg = auc(fpr_xg, tpr_xg)


    # PLOT THE ROC WITH AUC
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 1) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax0.minorticks_on()


    plt.plot(tpr_xg,1-fpr_xg,label="$f_{1}$, AUC = %0.2f" % roc_auc_xg,color="orange",ls=":",lw=4)
    plt.plot([1,0],[0,1],ls=":",color="grey")
    plt.xlabel("True Positive Rate",fontsize=16)
    plt.ylabel("1 $-$ False Positive Rate",fontsize=16)
    plt.legend()
    plt.title(r"XGBoost",fontsize=16)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.yticks([0.2,0.4,0.6,0.8,1.])
    plt.legend(frameon=False,fontsize=16)
    plt.show()
    
    return roc_auc_xg, y_pred_xg
    
  
  
  
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#

  
  
    
########################
# OBTAIN PDFs WITH KDE #
########################
   
def KDE_fitter(B_data, S_data, events_to_fit, bandwidth_space, set_kernel='epanechnikov'):

	# FIT THE PDFs OF 2 DATA SAMPLES WITH KDE
	# Inputs:
		# B_data -> 1st sample to extract the 1st PDF
		# S_data -> 2nd sample to extract the 2nd PDF
		# events_to_fit -> number of events used to extract the PDF (same for both samples), B_data[:events_to_fit], S_data[:events_to_fit]
		# bandwidth_space -> array, allowed bandwidth values that are going to be used. Example: bandwidth_space = np.logspace(-4.0, 0.05, 20)
		# set_kernel -> KDE kernel
		
	# Outputs:
		# kde_bkg -> PDF of the 1st sample
		# kde_sig -> PDF of the 2nd sample
		# norm_factor_SM -> Area of the 1st PDF (useful to normalize the PDF) 
		# norm_factor_NP -> Area of the 2nd PDF (useful to normalize the PDF) 
		# B_bandwidth -> bandwidth found/used for the 1st sample
		# S_bandwidth -> bandwidth found/used for the 2nd sample
    
    #############
    # Bandwidth #
    #############
    
    # for B_data
    kde = KernelDensity(kernel=set_kernel)
    grid = GridSearchCV(kde, {'bandwidth': bandwidth_space})
    grid.fit(np.c_[B_data[:events_to_fit]])

    B_bandwidth = grid.best_estimator_.bandwidth


    # for S_data
    kde = KernelDensity(kernel=set_kernel)
    grid = GridSearchCV(kde, {'bandwidth': bandwidth_space})
    grid.fit(np.c_[S_data[:events_to_fit]])

    S_bandwidth = grid.best_estimator_.bandwidth


    ########
    # PDFs #
    ########
    
    # with each bandwidth estimate the PDFs with KDE
    kde_bkg = KernelDensity(kernel=set_kernel, bandwidth=B_bandwidth).fit(np.c_[B_data, np.zeros(len(B_data)) ])
    kde_sig = KernelDensity(kernel=set_kernel, bandwidth=S_bandwidth).fit(np.c_[S_data, np.ones(len(S_data)) ])
    
    
    #################
    # Normalization #
    #################
    
    # range
    min_val = np.min([np.min(B_data),np.min(S_data)])
    max_val = np.max([np.max(B_data),np.max(S_data)])

    # points
    s_vals = np.linspace(min_val,max_val,1000)

    # evaluate the PDFs for each value of s
    dens_bkg = np.exp(kde_bkg.score_samples(np.c_[s_vals, np.zeros(len(s_vals)) ]) )
    dens_sig = np.exp(kde_sig.score_samples(np.c_[s_vals, np.ones(len(s_vals)) ]) )

    # Area
    norm_factor_SM = sum(dens_bkg*(s_vals[1]-s_vals[0]))
    norm_factor_NP = sum(dens_sig*(s_vals[1]-s_vals[0]))

    # normalize
    dens_bkg = dens_bkg / norm_factor_SM
    dens_sig = dens_sig / norm_factor_NP
    
    
    
    # plot to check the estimation
    plt.figure(figsize=(7,5))

    plt.hist(B_data, 25, range=[min_val,max_val], density=True, color='blue',alpha=0.5, linewidth=2, label=r'Binned B data');
    plt.hist(S_data, 25, range=[min_val,max_val], density=True, color='red',alpha=0.5, linewidth=2, label=r'Binned S data');

    plt.plot(s_vals, dens_bkg, color='blue',label=r'KDE B data',linestyle='dashed');
    plt.plot(s_vals, dens_sig, color='red',label=r'KDE S data',linestyle='dashed');

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.yscale('log')
    plt.xlabel("Variable",fontsize=14)
    plt.ylabel("Fraction of events/bin size",fontsize=14)
    plt.grid()
    plt.legend(loc='lower center',fontsize=13)
    plt.show()
    

    print('KDE Kernel: ', set_kernel)
    print('Background bandwidth: ', B_bandwidth)
    print('Signal bandwidth: ', S_bandwidth)
    
    return kde_bkg, kde_sig, norm_factor_SM, norm_factor_NP, B_bandwidth, S_bandwidth
    




#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#




##############################
# AVERAGE MAX NUMBER OF BINS #
##############################

def max_num_bins(B_data, B_expected, range_dat, MIN_EVS, bins_to_test):

	# FINDS THE MAXIMUN NUMBER OF LINEAR BINS THAT SATISFY THAT EACH BIN HAS A NUMBER OF EVENTS >= MIN_EVS
	# Inputs:
		# B_data -> data sample (can be as large as you want)
		# B_expected -> number of events that you want to bin, B_data[:N_events_back]
		# range_dat -> range of the data where you want to bin. Example range_dat=[[0,1]]
		# MIN_EVS -> minimum number of events that you allow in a bin
		# bins_to_test -> range of number of bins that you want to test. Example: bins_to_test=range(1,500)
		
	# Outputs:
		# max_bins -> maximum number of bins that satisfy the condition: # events per bin >= MIN_EVS
    
    # Les't find the number of possible ensembles
    num_pseudo_back = int(len(B_data) / B_expected)
    N_events_back = num_pseudo_back * B_expected


    for bin_it in bins_to_test:
        
        max_bins = bin_it
        
        # bin the parameter space of all background events
        hist_back, binedges_back = np.histogramdd([B_data[:N_events_back]], bins=(bin_it), range = range_dat)
        
        if min(hist_back) < MIN_EVS * num_pseudo_back:
            max_bins = bin_it - 1
            print('At least ' + str(MIN_EVS) + ' B events per bin, range = ' + str(range_dat) + ':')
            print('# bins: ', max_bins, 'OK')
            #print(min(hist_back/num_pseudo_back))
            break
            
    if min(hist_back) >= MIN_EVS * num_pseudo_back:
        print('At least ' + str(MIN_EVS) + ' B events per bin, range = ' + str(range_dat) + ':')
        print('# bins: ', max(bins_to_test), 'OK (max # bins tested)')

    return max_bins





#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#




################################
# SIGNIFICANCE, AZIMOV DATASET #
################################

def Z_BL_asimov(D_or_E, B_data, S_data, B_expected, S_expected, num_bins, range_dat, MIN_EVS, linear_bins=True):

	# FIND THE SIGNIFICANCE USING THE ASIMOV DATASET (no statistical error) WITH POISSON BIN LIKELIHOOD
	# Inputs:
		# D_or_E -> statistical test. Options: "exclusion" or "discovery"
		# B_data -> Background data sample
		# S_data -> Signal data sample
		# B_expected -> number of Background events expected in a pseudo experiment
		# S_expected -> number of Signal events expected in a pseudo experiment
		# num_bins -> number of bins
		# range_dat -> range of the data where you want to bin. Example: range_dat = [[0,1]]
		# MIN_EVS -> minimum number of events that you allow in a bin
		# linear_bins=True -> If True all the bins have the same width. If False all the bins have the same number of BACKGROUND events (but not the same bin width)
		
	# Outputs:
		# Z_bins -> value of the significance
    
    # Les't use an integer number of possible ensembles
    num_pseudo_back = int(len(B_data) / B_expected)
    N_events_back = num_pseudo_back * B_expected

    num_pseudo_sig = int(len(S_data) / S_expected)
    N_events_sig = num_pseudo_sig * S_expected
    
    
    #Let's find out the expected number of B and S events in each bin:
    
    if linear_bins==False: # same number of B events per bin
        bin_edges_same = pd.qcut(B_data[:N_events_back], q = num_bins, precision=0, retbins = True)[1]
        num_bins = [bin_edges_same]

    # bin the parameter space of all background events
    hist_back, binedges_back = np.histogramdd(B_data[:N_events_back], bins=num_bins, range = range_dat)
    # now divide by the number of possible ensembles
    back_prom = hist_back / num_pseudo_back

    # same for signal
    hist_sig, binedges_sig = np.histogramdd(S_data[:N_events_sig], bins=num_bins, range = range_dat)
    sig_prom = hist_sig / num_pseudo_sig
    
    
    # minimum number of background events per bin
    if min(back_prom) >= MIN_EVS:
        
        if D_or_E == 'exclusion':
            Z_bins = ( 2* sum( ( back_prom * np.log( back_prom/(sig_prom + back_prom) ) ) + sig_prom ) )**0.5
        elif D_or_E == 'discovery':
            Z_bins = ( 2* sum( (sig_prom+back_prom) * np.log( 1 + (sig_prom/back_prom) ) - sig_prom ) )**0.5
    
    else:
        raise ValueError('Minimum number of background events per bin condition NOT satisfied. Requested MIN_EVS=', MIN_EVS, ' but in data MIN_EVS=', min(back_prom))

    return Z_bins
    
    
    

#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#




###############################
# POISSON BIN LIKELIHOOD TEST #
###############################

def BL_test_fast(D_or_E, B_data, S_data, B_expected, S_expected, num_pseudo, num_bins, range_dat, MIN_EVS, linear_bins=True):

	# FIND THE SIGNIFICANCE WITH POISSON BIN LIKELIHOOD USING THE TEST STATISTIC FORMALISM (gives statistical error) 
	# Inputs:
		# D_or_E -> statistical test. Options: "exclusion" or "discovery"
		# B_data -> Background data sample
		# S_data -> Signal data sample
		# B_expected -> number of Background events expected in a pseudo experiment
		# S_expected -> number of Signal events expected in a pseudo experiment
		# num_pseudo -> number of pseudo experiments that are used to compute the test statistic distribution
		# num_bins -> number of bins
		# range_dat -> range of the data where you want to bin. Example: range_dat = [[0,1]]
		# MIN_EVS -> minimum number of events that you allow in a bin
		# linear_bins=True -> If True all the bins have the same width. If False all the bins have the same number of BACKGROUND events (but not the same bin width)
		
	# Outputs:
		# Z_bins -> value of the significance (using mu_hat that fits the data sample)
		# Z_bins_std -> statistical error (1 sigma) of the significance (using mu_hat that fits the data sample)
		# muhat_mean_bins -> mean of the mu_hat computed from the data sample
		# Z_bins_mu -> value of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
		# Z_bins_std_mu -> statistical error (1 sigma) of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
    
    if linear_bins==False: # same number of B events per bin
        bin_edges_same = pd.qcut(B_data, q = num_bins, precision=0, retbins = True)[1]
        num_bins = [bin_edges_same]
        
        
    # Expected background events per bin ,cömo
    hist_back, binedges_back = np.histogramdd([B_data], bins=(num_bins), range = range_dat)
    bin_edges = binedges_back[0]
    back_prom_bins = hist_back / (len(B_data) / B_expected)

    # find the minimum
    back_prom_noceros = []
    for i in range(len(back_prom_bins)):
        if back_prom_bins[i]!=0:
            back_prom_noceros.append(back_prom_bins[i])

    min_back = min(back_prom_noceros)

    # replace the zeros
    for i in range(len(back_prom_bins)):
        if back_prom_bins[i]==0:
            back_prom_bins[i] = min_back
        
        
    
    # Expected signal events per bin
    hist_sig, binedges_sig = np.histogramdd([S_data], bins=(num_bins), range = range_dat)
    sig_prom_bins = hist_sig / (len(S_data) / S_expected)
    
    


    muhat_selected_bins_list = []
    q_muhat_bins = []
    q_muhat_bins_mu = []
    
    # to check how many pseudo experiments do not satisfy the MIN_EVS condition
    fail_pseudo = 0
    
    # loop over the number of pseudo experiments
    for its in range(num_pseudo):
        B_rand = np.random.poisson(int(B_expected))
        
        B_data_shuf = np.random.choice(B_data, size = B_rand, replace = False)
        
        
        try:
            if D_or_E == 'exclusion': # only Background events
                
                pseudo_exp = B_data_shuf # only B
    
                
                # Let's find out the number of B and S events in each bin for this pseudo-experiment:
    
                # bin the parameter space of all background events
                hist_N, binedges_N = np.histogramdd([pseudo_exp], bins=(num_bins), range = range_dat)
                # events per bin of the pseudo experiment
                N_pseudo = hist_N
    
                if min(N_pseudo) >= MIN_EVS:
                    
                    # approximation: mu_hat=0
                    q_muhat_bins_mu.append( -2 * sum([( (ni * np.log( ((1.*si)+bi)/((0.*si)+bi) ) ) - ((1.*si)+bi) + ((0.*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )
    
                    
                    # reference points
                    sum_muhat_zero = sum ( [((ni*si) / (((-0.25)*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
                    sum_muhat_one = sum ( [((ni*si) / ((1.*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
    
    
                    # we need (implicit eq. for mu_hat) = 0
                    # conditions considering the reference points
                    if (sum_muhat_zero < sum_muhat_one < 0) or (0 < sum_muhat_one < sum_muhat_zero):
    
                        muhat_selected_bins = 1.1
    
                    elif (sum_muhat_one < sum_muhat_zero < 0) or (0 < sum_muhat_zero < sum_muhat_one):
    
                        muhat_selected_bins = -0.3
    
                    elif sum_muhat_zero < 0 < sum_muhat_one:
    
                        # grid, mu_hat is around 0
                        muhat_test = np.arange(-0.25, 1., 0.05)
    
                        for vv in range(len(muhat_test)):
    
                            mu_hat_condition_equal_0 = sum ( [((ni*si) / ((muhat_test[vv]*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
                            
                            if mu_hat_condition_equal_0 > 0:
                                muhat_selected_bins = muhat_test[vv]
                                break
    
                    elif sum_muhat_one < 0 < sum_muhat_zero:
    
                        # grid, mu_hat is around 0
                        muhat_test = np.arange(-0.25, 1., 0.05)
    
                        for vv in range(len(muhat_test)):
    
                            mu_hat_condition_equal_0 = sum ( [((ni*si) / ((muhat_test[vv]*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
    
                            if mu_hat_condition_equal_0 < 0:
                                muhat_selected_bins = muhat_test[vv]
                                break
                            
                            
                    # save the computed mu_hat for each pseudo_experiment
                    muhat_selected_bins_list.append(muhat_selected_bins)
    
                    
                    # compute the test statistic for each pseudo_exp considering mu_hat
                    if muhat_selected_bins > 1:
                        q_muhat_bins.append( 0 )
                        
                    elif muhat_selected_bins > 0:
                        q_muhat_bins.append( -2 * sum([( (ni * np.log( ((1.*si)+bi)/((muhat_selected_bins*si)+bi) ) ) - ((1.*si)+bi) + ((muhat_selected_bins*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )
    
                    else:
                        q_muhat_bins.append( -2 * sum([( (ni * np.log( ((1.*si)+bi)/((0.*si)+bi) ) ) - ((1.*si)+bi) + ((0.*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )
    
                else:
                    fail_pseudo += 1
                    
                    
                
            if D_or_E == 'discovery': # Background and Signal events
                
                S_rand = np.random.poisson(int(S_expected))
                
                S_data_shuf = np.random.choice(S_data, size = S_rand, replace = False)
                
                
                pseudo_exp = np.concatenate([B_data_shuf,S_data_shuf]) # Background and Signal
                
    
    
                # Let's find out the expected number of B and S events in each bin:
    
                # bin the parameter space of all background events
                hist_N, binedges_N = np.histogramdd([pseudo_exp], bins=(num_bins), range = range_dat)
                # events per bin of the pseudo experiment
                N_pseudo = hist_N
    
                if min(N_pseudo) >= MIN_EVS:
                    
                    # approximation: mu_hat=1
                    q_muhat_bins_mu.append( -2 * sum([( (ni * np.log( ((0.*si)+bi)/((1.*si)+bi) ) ) - ((0.*si)+bi) + ((1.*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )
    
                    
                    # reference points
                    sum_muhat_zero = sum ( [((ni*si) / (((0)*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
                    sum_muhat_two = sum ( [((ni*si) / ((2.*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
    
                    
                    # we need (implicit eq. for mu_hat) = 0
                    # conditions considering the reference points
                    if (sum_muhat_zero < sum_muhat_two < 0) or (0 < sum_muhat_two < sum_muhat_zero):
    
                        muhat_selected_bins = 2.1
    
                    elif (sum_muhat_two < sum_muhat_zero < 0) or (0 < sum_muhat_zero < sum_muhat_two):
    
                        muhat_selected_bins = -0.1
    
                    elif sum_muhat_zero < 0 < sum_muhat_two:
    
                        # grid, mu_hat is around 1
                        muhat_test = np.arange(0, 2., 0.05)
    
                        for vv in range(len(muhat_test)):
    
                            mu_hat_condition_equal_0 = sum ( [((ni*si) / ((muhat_test[vv]*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
    
                            if mu_hat_condition_equal_0 > 0:
                                muhat_selected_bins = muhat_test[vv]
                                break
    
                    elif sum_muhat_two < 0 < sum_muhat_zero:
    
                        # grid, mu_hat is around 1
                        muhat_test = np.arange(0, 2., 0.05)
    
                        for vv in range(len(muhat_test)):
    
                            mu_hat_condition_equal_0 = sum ( [((ni*si) / ((muhat_test[vv]*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )
    
                            if mu_hat_condition_equal_0 < 0:
                                muhat_selected_bins = muhat_test[vv]
                                break
    
    
                    # save the computed mu_hat for each pseudo_experiment
                    muhat_selected_bins_list.append(muhat_selected_bins)
                    
                    
                    # compute the test statistic for each pseudo_exp considering mu_hat
                    if muhat_selected_bins > 0:
                        q_muhat_bins.append( -2 * sum([( (ni * np.log( ((0.*si)+bi)/((muhat_selected_bins*si)+bi) ) ) - ((0.*si)+bi) + ((muhat_selected_bins*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )
                    
                    else:
                        q_muhat_bins.append( 0 )
                        
                else:
                    fail_pseudo += 1
        except:
            fail_pseudo += 1
        
                
                
                
    print('Ratio of pseudo experiments that do not satisfied the MIN_EVS condition: ', fail_pseudo/num_pseudo)
    
    # Histogram of q_muhats
    plt.figure(figsize=(7,5))
    
    weights = np.ones_like(q_muhat_bins)/float(len(q_muhat_bins))
    plt.hist(q_muhat_bins, 25, weights=weights, histtype='step', color='blue', linewidth=2, label=r'$\hat{\mu}$ from data')
    
    weights = np.ones_like(q_muhat_bins_mu)/float(len(q_muhat_bins_mu))
    if D_or_E == 'exclusion':
        plt.hist(q_muhat_bins_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=0$')
    if D_or_E == 'discovery':
        plt.hist(q_muhat_bins_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=1$')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("$q$",fontsize=16)
    plt.ylabel("Fraction of pseudo experiments",fontsize=16)
    plt.grid()
    plt.legend(fontsize=14)
    plt.show()
    
    
    
    # With the calculation of mu_hat
    muhat_mean_bins = np.mean(muhat_selected_bins_list)

    Z_bins = abs( np.median(q_muhat_bins) )**0.5
    Z_bins_std = np.std(q_muhat_bins) / (2.*Z_bins)
    
    
    # With the approximation of mu_hat
    Z_bins_mu = abs( np.median(q_muhat_bins_mu) )**0.5
    Z_bins_std_mu = np.std(q_muhat_bins_mu) / (2.*Z_bins_mu)
    
    
    return Z_bins, Z_bins_std, muhat_mean_bins, Z_bins_mu, Z_bins_std_mu
    


    
#-----------------------------------------------------#


    
    
def BL_test_fsolve(D_or_E, B_data, S_data, B_expected, S_expected, num_pseudo, num_bins, range_dat, MIN_EVS, linear_bins=True):

	# FIND THE SIGNIFICANCE WITH POISSON BIN LIKELIHOOD USING THE TEST STATISTIC FORMALISM (gives statistical error) 
	# Inputs:
		# D_or_E -> statistical test. Options: "exclusion" or "discovery"
		# B_data -> Background data sample
		# S_data -> Signal data sample
		# B_expected -> number of Background events expected in a pseudo experiment
		# S_expected -> number of Signal events expected in a pseudo experiment
		# num_pseudo -> number of pseudo experiments that are used to compute the test statistic distribution
		# num_bins -> number of bins
		# range_dat -> range of the data where you want to bin. Example: range_dat = [[0,1]]
		# MIN_EVS -> minimum number of events that you allow in a bin
		# linear_bins=True -> If True all the bins have the same width. If False all the bins have the same number of BACKGROUND events (but not the same bin width)
		
	# Outputs:
		# Z_bins -> value of the significance (using mu_hat that fits the data sample)
		# Z_bins_std -> statistical error (1 sigma) of the significance (using mu_hat that fits the data sample)
		# muhat_mean_bins -> mean of the mu_hat computed from the data sample
		# Z_bins_mu -> value of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
		# Z_bins_std_mu -> statistical error (1 sigma) of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
    
    if linear_bins==False: # same number of B events per bin
        bin_edges_same = pd.qcut(B_data, q = num_bins, precision=0, retbins = True)[1]
        num_bins = [bin_edges_same]
        
        
    # Expected background events per bin ,cömo
    hist_back, binedges_back = np.histogramdd([B_data], bins=(num_bins), range = range_dat)
    bin_edges = binedges_back[0]
    back_prom_bins = hist_back / (len(B_data) / B_expected)

    # find the minimum
    back_prom_noceros = []
    for i in range(len(back_prom_bins)):
        if back_prom_bins[i]!=0:
            back_prom_noceros.append(back_prom_bins[i])

    min_back = min(back_prom_noceros)

    # replace the zeros
    for i in range(len(back_prom_bins)):
        if back_prom_bins[i]==0:
            back_prom_bins[i] = min_back
        
        
    
    # Expected signal events per bin
    hist_sig, binedges_sig = np.histogramdd([S_data], bins=(num_bins), range = range_dat)
    sig_prom_bins = hist_sig / (len(S_data) / S_expected)
    
    


    muhat_selected_bins_list = []
    q_muhat_bins = []
    q_muhat_bins_mu = []
    
    # to check how many pseudo experiments do not satisfy the MIN_EVS condition
    fail_pseudo = 0
    
    # loop over the number of pseudo experiments
    for its in range(num_pseudo):
        
        B_rand = np.random.poisson(int(B_expected))
        
        B_data_shuf = np.random.choice(B_data, size = B_rand, replace = False)
        
        
        if D_or_E == 'exclusion': # only Background events
            
            pseudo_exp = B_data_shuf # only B

            
            # Let's find out the number of B and S events in each bin for this pseudo-experiment:

            # bin the parameter space of all background events
            hist_N, binedges_N = np.histogramdd([pseudo_exp], bins=(num_bins), range = range_dat)
            # events per bin of the pseudo experiment
            N_pseudo = hist_N

            if min(N_pseudo) >= MIN_EVS:
                
                # approximation: mu_hat=0
                q_muhat_bins_mu.append( -2 * sum([( (ni * np.log( ((1.*si)+bi)/((0.*si)+bi) ) ) - ((1.*si)+bi) + ((0.*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )

                
                # compute mu_hat from implicit equation
                def f_mu_hat(mu_h, N_pseudo=N_pseudo, sig_prom_bins=sig_prom_bins, back_prom_bins=back_prom_bins):
                    return sum ( [((ni*si) / ((mu_h*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )

        
                muhat_selected_bins = fsolve(f_mu_hat,0)[0]
            
                # save the computed mu_hat for each pseudo_experiment
                muhat_selected_bins_list.append(muhat_selected_bins)

                
                # compute the test statistic for each pseudo_exp considering mu_hat
                if muhat_selected_bins > 1:
                    q_muhat_bins.append( 0 )
                    
                elif muhat_selected_bins > 0:
                    q_muhat_bins.append( -2 * sum([( (ni * np.log( ((1.*si)+bi)/((muhat_selected_bins*si)+bi) ) ) - ((1.*si)+bi) + ((muhat_selected_bins*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )

                else:
                    q_muhat_bins.append( -2 * sum([( (ni * np.log( ((1.*si)+bi)/((0.*si)+bi) ) ) - ((1.*si)+bi) + ((0.*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )

            else:
                fail_pseudo += 1
                
                
            
        if D_or_E == 'discovery': # Background and Signal events
            
            S_rand = np.random.poisson(int(S_expected))
            
            S_data_shuf = np.random.choice(S_data, size = S_rand, replace = False)
            
            
            pseudo_exp = np.concatenate([B_data_shuf,S_data_shuf]) # Background and Signal
            


            # Let's find out the expected number of B and S events in each bin:

            # bin the parameter space of all background events
            hist_N, binedges_N = np.histogramdd([pseudo_exp], bins=(num_bins), range = range_dat)
            # events per bin of the pseudo experiment
            N_pseudo = hist_N

            if min(N_pseudo) >= MIN_EVS:
                
                # approximation: mu_hat=1
                q_muhat_bins_mu.append( -2 * sum([( (ni * np.log( ((0.*si)+bi)/((1.*si)+bi) ) ) - ((0.*si)+bi) + ((1.*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )

                
                # compute mu_hat from implicit equation
                def f_mu_hat(mu_h, N_pseudo=N_pseudo, sig_prom_bins=sig_prom_bins, back_prom_bins=back_prom_bins):
                    return sum ( [((ni*si) / ((mu_h*si) + bi)) - si for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)] )

                
                muhat_selected_bins = fsolve(f_mu_hat,1)[0]
            
                # save the computed mu_hat for each pseudo_experiment
                muhat_selected_bins_list.append(muhat_selected_bins)
                
                
                # compute the test statistic for each pseudo_exp considering mu_hat
                if muhat_selected_bins > 0:
                    q_muhat_bins.append( -2 * sum([( (ni * np.log( ((0.*si)+bi)/((muhat_selected_bins*si)+bi) ) ) - ((0.*si)+bi) + ((muhat_selected_bins*si)+bi) ) for ni, si, bi in zip(N_pseudo, sig_prom_bins, back_prom_bins)]) )
                
                else:
                    q_muhat_bins.append( 0 )
                    
            else:
                fail_pseudo += 1
                
                    
    print('Ratio of pseudo experiments that do not satisfied the MIN_EVS condition: ', fail_pseudo/num_pseudo)
                
    # Histogram of q_muhats
    plt.figure(figsize=(7,5))
    
    weights = np.ones_like(q_muhat_bins)/float(len(q_muhat_bins))
    plt.hist(q_muhat_bins, 25, weights=weights, histtype='step', color='blue', linewidth=2, label=r'$\hat{\mu}$ from data')
    
    weights = np.ones_like(q_muhat_bins_mu)/float(len(q_muhat_bins_mu))
    if D_or_E == 'exclusion':
        plt.hist(q_muhat_bins_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=0$')
    if D_or_E == 'discovery':
        plt.hist(q_muhat_bins_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=1$')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("$q$",fontsize=16)
    plt.ylabel("Fraction of pseudo experiments",fontsize=16)
    plt.grid()
    plt.legend(fontsize=14)
    plt.show()
    
    
    
    # With the calculation of mu_hat
    muhat_mean_bins = np.mean(muhat_selected_bins_list)

    Z_bins = abs( np.median(q_muhat_bins) )**0.5
    Z_bins_std = np.std(q_muhat_bins) / (2.*Z_bins)
    
    
    # With the approximation of mu_hat
    Z_bins_mu = abs( np.median(q_muhat_bins_mu) )**0.5
    Z_bins_std_mu = np.std(q_muhat_bins_mu) / (2.*Z_bins_mu)
    
    
    return Z_bins, Z_bins_std, muhat_mean_bins, Z_bins_mu, Z_bins_std_mu

    
    
    

#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#




################################
# SIGNIFICANCE, AZIMOV DATASET #
################################

def MLL_test_fast(D_or_E, pB_B_data, pS_B_data, pB_S_data, pS_S_data, B_expected, S_expected, num_pseudo):

	# FIND THE SIGNIFICANCE WITH MLL USING THE TEST STATISTIC FORMALISM (gives statistical error) 
	# Inputs:
		# D_or_E -> statistical test. Options: "exclusion" or "discovery"
		# pB_B_data -> array with BACKGROUND PDF evaluated in each BACKGROUND event,   p_B(background),   shape (len(background), )
		# pS_B_data -> array with SIGNAL PDF evaluated in each BACKGROUND event,       p_B(background),   shape (len(background), )
		# pB_S_data -> array with BACKGROUND PDF evaluated in each SIGNAL event,       p_B(background),   shape (len(signal), )
		# pS_S_data -> array with SIGNAL PDF evaluated in each SIGNAL event,           p_B(background),   shape (len(signal), )
		# B_expected -> number of Background events expected in a pseudo experiment
		# S_expected -> number of Signal events expected in a pseudo experiment
		# num_pseudo -> number of pseudo experiments that are used to compute the test statistic distribution
		
	# Outputs:
		# Z_bins -> value of the significance (using mu_hat that fits the data sample)
		# Z_bins_std -> statistical error (1 sigma) of the significance (using mu_hat that fits the data sample)
		# muhat_mean_bins -> mean of the mu_hat computed from the data sample
		# Z_bins_mu -> value of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
		# Z_bins_std_mu -> statistical error (1 sigma) of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
    
    # indeces (to later build a random pseudo experiment)
    indices_B = [i for i in range(len(pS_B_data))]

    if D_or_E == 'discovery':
        indices_S = [i for i in range(len(pS_S_data))]
        
        
    muhat_selected_MLL_list = []
    q_muhat_MLL = []
    q_muhat_MLL_mu = []
    
    # loop over the number of pseudo experiments
    
    fail_pseudo = 0
    for its in range(num_pseudo):
        
        # this pseudo-exp has B_rand number of B events
        B_rand = np.random.poisson(int(B_expected))

        ran_ind = np.random.choice(indices_B, B_rand)

        pB_B_data_shuf = []
        pS_B_data_shuf = []

        # for each event x_i in the pseudo, save pb(o(x_i)) and ps(o(x_i)) (notice its the same x_i for pb and ps)
        for i in ran_ind:
            pB_B_data_shuf.append(pB_B_data[i])
            pS_B_data_shuf.append(pS_B_data[i])

        pB_B_data_shuf  = np.array(pB_B_data_shuf)
        pS_B_data_shuf  = np.array(pS_B_data_shuf)
        
        try:
            if D_or_E == 'exclusion': # only Background events
            
                # p_b(o(x_ensemble)) =  p_b(o(B_ensemble))
                prob_x_given_B = pB_B_data_shuf
    
                # p_s(o(x_ensemble)) =  p_s(o(B_ensemble))
                prob_x_given_S = pS_B_data_shuf
    
                if np.min(prob_x_given_B) == 0:
                    print('There are events with p(s)=0')
                    prob_x_given_B[np.where(prob_x_given_B == 0 )[0]] = np.min(prob_x_given_B[np.where(prob_x_given_B > 0 )[0]])
    
                # NOW WE HAVE p_{s,b}(x_ensemble) for this particular pseudo_experiment
    
                # approximation: mu_hat=0 (exclusion)
                q_muhat_MLL_mu.append( 2 * ( ( (1.-0.) * S_expected ) - sum( [np.log( ( (B_expected*y) + (S_expected*x) ) / ( (B_expected*y) + (0.*S_expected*x) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )
    
                
                # ESTIMATE mu_hat for this particular ensemble (implicit equation)
                B_prob_x_given_B = [x * B_expected for x in prob_x_given_B]
                
                # reference points
                sum_muhat_zero = sum ( [(x*1.) / ( (x * (-0.25) * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
                sum_muhat_one = sum ( [(x*1.) / ( (x * 1. * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
    
                
                # we need (implicit eq. for mu_hat) = 1
                # conditions considering the reference points
                if (sum_muhat_zero < sum_muhat_one < 1) or (1 < sum_muhat_one < sum_muhat_zero):
                    
                    muhat_selected_MLL = 1.1
                    
                elif (sum_muhat_one < sum_muhat_zero < 1) or (1 < sum_muhat_zero < sum_muhat_one):
                    
                    muhat_selected_MLL = -0.3
    
                elif sum_muhat_zero < 1 < sum_muhat_one:
                    
                    # grid, mu_hat is around 0
                    muhat_test = np.arange(-0.25, 1., 0.05)
    
                    for vv in range(len(muhat_test)):
    
                        mu_hat_condition_equal_1 = sum ( [(x*1.) / ( (x * muhat_test[vv] * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
    
                        if mu_hat_condition_equal_1 > 1:
                            muhat_selected_MLL = muhat_test[vv]
                            break
    
                elif sum_muhat_one < 1 < sum_muhat_zero:
                    
                    # grid, mu_hat is around 0
                    muhat_test = np.arange(-0.25, 1., 0.05)
    
                    for vv in range(len(muhat_test)):
    
                        mu_hat_condition_equal_1 = sum ( [(x*1.) / ( (x * muhat_test[vv] * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
    
                        if mu_hat_condition_equal_1 < 1:
                            muhat_selected_MLL = muhat_test[vv]
                            break
                            
                            
                # save the computed mu_hat (within range) for each pseudo_experiment
                muhat_selected_MLL_list.append(muhat_selected_MLL)
                            
                            
                # compute the test statistic for each pseudo_exp considering mu_hat
                if muhat_selected_MLL > 1:
                    q_muhat_MLL.append( 0 )
    
                elif muhat_selected_MLL > 0:
                    q_muhat_MLL.append( 2 * ( ( (1.-muhat_selected_MLL) * S_expected ) - sum( [np.log( ( (B_expected*y) + (S_expected*x) ) / ( (B_expected*y) + (muhat_selected_MLL*S_expected*x) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )
    
                else:
                    q_muhat_MLL.append( 2 * ( ( (1.-0.) * S_expected ) - sum( [np.log( ( (B_expected*y) + (S_expected*x) ) / ( (B_expected*y) + (0*S_expected*x) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )
    
                
                
                
                
    
    
            if D_or_E == 'discovery': # Background and Signal events
                
                # this pseudo-exp has S_rand number of S events
                S_rand = np.random.poisson(int(S_expected))
                
                ran_ind = np.random.choice(indices_S, S_rand)
    
                pB_S_data_shuf = []
                pS_S_data_shuf = []
    
                # for each event x_i in the pseudo, save pb(o(x_i)) and ps(o(x_i)) (notice its the same x_i for pb and ps)
                for i in ran_ind:
                    pB_S_data_shuf.append(pB_S_data[i])
                    pS_S_data_shuf.append(pS_S_data[i])
    
                pB_S_data_shuf  = np.array(pB_S_data_shuf)
                pS_S_data_shuf  = np.array(pS_S_data_shuf)
                
                
                # p_b(o(x_ensemble)) =  concatenate: p_b(o(B_ensemble)) and p_b(o(S_ensemble)) 
                prob_x_given_B = np.concatenate([pB_B_data_shuf,pB_S_data_shuf])
    
                # p_s(o(x_ensemble)) =  concatenate: p_s(o(B_ensemble)) and p_s(o(S_ensemble)) 
                prob_x_given_S = np.concatenate([pS_B_data_shuf,pS_S_data_shuf])
    
    
                if np.min(prob_x_given_B) == 0:
                    print('There are events with p(s)=0')
                    prob_x_given_B[np.where(prob_x_given_B == 0 )[0]] = np.min(prob_x_given_B[np.where(prob_x_given_B > 0 )[0]])
                    print(np.min(prob_x_given_B))
    
                # NOW WE HAVE p_{s,b}(x_ensemble) for this particular pseudo_experiment
    
                # approximation: mu_hat=1 (discovery)
                q_muhat_MLL_mu.append( 2 * ( ( -1. * S_expected) + sum( [np.log( 1 + ( (1.*S_expected/B_expected) * (x / y) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )
    
                
                
                # ESTIMATE mu_hat for this particular ensemble (implicit equation)
                B_prob_x_given_B = [x * B_expected for x in prob_x_given_B]
                
                # reference points
                sum_muhat_zero = sum ( [(x*1.) / ( (x * (0) * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
                sum_muhat_two = sum ( [(x*1.) / ( (x * 2. * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
    
                
                # we need (implicit eq. for mu_hat) = 1
                # conditions considering the reference points
                if (sum_muhat_zero < sum_muhat_two < 1) or (1 < sum_muhat_two < sum_muhat_zero):
                    
                    muhat_selected_MLL = 2.1
                    
                elif (sum_muhat_two < sum_muhat_zero < 1) or (1 < sum_muhat_zero < sum_muhat_two):
                    
                    muhat_selected_MLL = -0.1
    
                elif sum_muhat_zero < 1 < sum_muhat_two:
                    
                    # grid, mu_hat is around 1
                    muhat_test = np.arange(0, 2.05, 0.05)
    
                    for vv in range(len(muhat_test)):
    
                        mu_hat_condition_equal_1 = sum ( [(x*1.) / ( (x * muhat_test[vv] * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
    
                        if mu_hat_condition_equal_1 > 1:
                            muhat_selected_MLL = muhat_test[vv]
                            break
    
                elif sum_muhat_two < 1 < sum_muhat_zero:
                    
                    # grid, mu_hat is around 1
                    muhat_test = np.arange(0, 2.05, 0.05)
    
                    for vv in range(len(muhat_test)):
    
                        mu_hat_condition_equal_1 = sum ( [(x*1.) / ( (x * muhat_test[vv] * S_expected) + y ) for x, y in zip(prob_x_given_S, B_prob_x_given_B)] )
    
                        if mu_hat_condition_equal_1 < 1:
                            muhat_selected_MLL = muhat_test[vv]
                            break
                            
                            
                # save the computed mu_hat (within range) for each pseudo_experiment
                if 'muhat_selected_MLL' not in locals():
                    print('muhat2', sum_muhat_two)
                    print('muhat0', sum_muhat_zero)
                muhat_selected_MLL_list.append(muhat_selected_MLL)
                
                
                # compute the test statistic for each pseudo_exp considering mu_hat
                if muhat_selected_MLL > 0:
                    q_muhat_MLL.append( 2 * ( (-1.*muhat_selected_MLL * S_expected) + sum( [np.log( 1 + ( (muhat_selected_MLL*S_expected/B_expected) * (x / y) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )
    
                else:
                    q_muhat_MLL.append( 0 )
        except:
            fail_pseudo += 1
                
            
    print('Ratio of pseudo experiments that do not satisfied the MIN_EVS condition: ', fail_pseudo/num_pseudo)
    # Histogram of q_muhats
    plt.figure(figsize=(7,5))
    
    weights = np.ones_like(q_muhat_MLL)/float(len(q_muhat_MLL))
    plt.hist(q_muhat_MLL, 25, weights=weights, histtype='step', color='blue', linewidth=2, label=r'$\hat{\mu}$ from data')
    
    weights = np.ones_like(q_muhat_MLL_mu)/float(len(q_muhat_MLL_mu))
    if D_or_E == 'exclusion':
        plt.hist(q_muhat_MLL_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=0$')
    if D_or_E == 'discovery':
        plt.hist(q_muhat_MLL_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=1$')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("$q$",fontsize=16)
    plt.ylabel("Fraction of pseudo experiments",fontsize=16)
    plt.grid()
    plt.legend(fontsize=14)
    plt.show()
    
    
    
    # With the calculation of mu_hat
    muhat_mean_MLL = np.mean(muhat_selected_MLL_list)

    Z_MLL = abs( np.median(q_muhat_MLL) )**0.5
    Z_MLL_std = np.std(q_muhat_MLL) / (2.*Z_MLL)
    
    
    # With the approximation of mu_hat
    Z_MLL_mu = abs( np.median(q_muhat_MLL_mu) )**0.5
    Z_MLL_std_mu = np.std(q_muhat_MLL_mu) / (2.*Z_MLL_mu)
    
    
    return Z_MLL, Z_MLL_std, muhat_mean_MLL, Z_MLL_mu, Z_MLL_std_mu




#-----------------------------------------------------#




def MLL_test_fsolve(D_or_E, pB_B_data, pS_B_data, pB_S_data, pS_S_data, B_expected, S_expected, num_pseudo):

	# FIND THE SIGNIFICANCE WITH MLL USING THE TEST STATISTIC FORMALISM (gives statistical error) 
	# Inputs:
		# D_or_E -> statistical test. Options: "exclusion" or "discovery"
		# pB_B_data -> array with BACKGROUND PDF evaluated in each BACKGROUND event,   p_B(background),   shape (len(background), )
		# pS_B_data -> array with SIGNAL PDF evaluated in each BACKGROUND event,       p_B(background),   shape (len(background), )
		# pB_S_data -> array with BACKGROUND PDF evaluated in each SIGNAL event,       p_B(background),   shape (len(signal), )
		# pS_S_data -> array with SIGNAL PDF evaluated in each SIGNAL event,           p_B(background),   shape (len(signal), )
		# B_expected -> number of Background events expected in a pseudo experiment
		# S_expected -> number of Signal events expected in a pseudo experiment
		# num_pseudo -> number of pseudo experiments that are used to compute the test statistic distribution
		
	# Outputs:
		# Z_bins -> value of the significance (using mu_hat that fits the data sample)
		# Z_bins_std -> statistical error (1 sigma) of the significance (using mu_hat that fits the data sample)
		# muhat_mean_bins -> mean of the mu_hat computed from the data sample
		# Z_bins_mu -> value of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
		# Z_bins_std_mu -> statistical error (1 sigma) of the significance (using a FIXED mu_hat=0 for exclusion, mu_hat=1 for discovery)
    
    # indeces (to later build a random pseudo experiment)
    indices_B = [i for i in range(len(pS_B_data))]

    if D_or_E == 'discovery':
        indices_S = [i for i in range(len(pS_S_data))]
        
        
    muhat_selected_MLL_list = []
    q_muhat_MLL = []
    q_muhat_MLL_mu = []
    
    # loop over the number of pseudo experiments
    for its in range(num_pseudo):
        
        # this pseudo-exp has B_rand number of B events
        B_rand = np.random.poisson(int(B_expected))

        ran_ind = np.random.choice(indices_B, B_rand)

        pB_B_data_shuf = []
        pS_B_data_shuf = []

        # for each event x_i in the pseudo, save pb(o(x_i)) and ps(o(x_i)) (notice its the same x_i for pb and ps)
        for i in ran_ind:
            pB_B_data_shuf.append(pB_B_data[i])
            pS_B_data_shuf.append(pS_B_data[i])

        pB_B_data_shuf  = np.array(pB_B_data_shuf)
        pS_B_data_shuf  = np.array(pS_B_data_shuf)
        
        
        if D_or_E == 'exclusion': # only Background events
        
            # p_b(o(x_ensemble)) =  p_b(o(B_ensemble))
            prob_x_given_B = pB_B_data_shuf

            # p_s(o(x_ensemble)) =  p_s(o(B_ensemble))
            prob_x_given_S = pS_B_data_shuf



            # NOW WE HAVE p_{s,b}(x_ensemble) for this particular pseudo_experiment

            # approximation: mu_hat=0 (exclusion)
            q_muhat_MLL_mu.append( 2 * ( ( (1.-0.) * S_expected ) - sum( [np.log( ( (B_expected*y) + (S_expected*x) ) / ( (B_expected*y) + (0.*S_expected*x) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )


            # compute mu_hat from implicit equation
            def f_mu_hat(mu_h, prob_x_given_B=prob_x_given_B, prob_x_given_S=prob_x_given_S):
                return sum ( [x / ( (x * mu_h * S_expected) + (y * B_expected) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) - 1


            muhat_selected_MLL = fsolve(f_mu_hat,0, xtol=1e-2)[0]

            # save the computed mu_hat for each pseudo_experiment
            muhat_selected_MLL_list.append(muhat_selected_MLL)
                        
                        
            # compute the test statistic for each pseudo_exp considering mu_hat
            if muhat_selected_MLL > 1:
                q_muhat_MLL.append( 0 )

            elif muhat_selected_MLL > 0:
                q_muhat_MLL.append( 2 * ( ( (1.-muhat_selected_MLL) * S_expected ) - sum( [np.log( ( (B_expected*y) + (S_expected*x) ) / ( (B_expected*y) + (muhat_selected_MLL*S_expected*x) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )

            else:
                q_muhat_MLL.append( 2 * ( ( (1.-0.) * S_expected ) - sum( [np.log( ( (B_expected*y) + (S_expected*x) ) / ( (B_expected*y) + (0*S_expected*x) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )

                    
  
            
        if D_or_E == 'discovery': # Background and Signal events
            
            S_rand = np.random.poisson(int(S_expected))
            
            ran_ind = np.random.choice(indices_S, S_rand)

            pB_S_data_shuf = []
            pS_S_data_shuf = []

            for i in ran_ind:
                pB_S_data_shuf.append(pB_S_data[i])
                pS_S_data_shuf.append(pS_S_data[i])

            pB_S_data_shuf  = np.array(pB_S_data_shuf)
            pS_S_data_shuf  = np.array(pS_S_data_shuf)
            
            
            # p_b(o(x_ensemble)) =  concatenate: p_b(o(B_ensemble)) and p_b(o(S_ensemble)) 
            prob_x_given_B = np.concatenate([pB_B_data_shuf,pB_S_data_shuf])

            # p_s(o(x_ensemble)) =  concatenate: p_s(o(B_ensemble)) and p_s(o(S_ensemble)) 
            prob_x_given_S = np.concatenate([pS_B_data_shuf,pS_S_data_shuf])



            # NOW WE HAVE p_{s,b}(x_ensemble) for this particular pseudo_experiment

            # approximation: mu_hat=1
            q_muhat_MLL_mu.append( 2 * ( ( -1. * S_expected) + sum( [np.log( 1 + ( (1.*S_expected/B_expected) * (x / y) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )


            # compute mu_hat from implicit equation
            def f_mu_hat(mu_h, prob_x_given_B=prob_x_given_B, prob_x_given_S=prob_x_given_S):
                return sum ( [x / ( (x * mu_h * S_expected) + (y * B_expected) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) - 1


            muhat_selected_MLL = fsolve(f_mu_hat,1, xtol=1e-2)[0]

            # save the computed mu_hat for each pseudo_experiment
            muhat_selected_MLL_list.append(muhat_selected_MLL)
            
            
            # compute the test statistic for each pseudo_exp considering mu_hat
            if muhat_selected_MLL > 0:
                q_muhat_MLL.append( 2 * ( (-1.*muhat_selected_MLL * S_expected) + sum( [np.log( 1 + ( (muhat_selected_MLL*S_expected/B_expected) * (x / y) ) ) for x, y in zip(prob_x_given_S, prob_x_given_B)] ) ) )

            else:
                q_muhat_MLL.append( 0 )

                
                
    # Histogram of q_muhats
    plt.figure(figsize=(7,5))
    
    weights = np.ones_like(q_muhat_MLL)/float(len(q_muhat_MLL))
    plt.hist(q_muhat_MLL, 25, weights=weights, histtype='step', color='blue', linewidth=2, label=r'$\hat{\mu}$ from data')
    
    weights = np.ones_like(q_muhat_MLL_mu)/float(len(q_muhat_MLL_mu))
    if D_or_E == 'exclusion':
        plt.hist(q_muhat_MLL_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=0$')
    if D_or_E == 'discovery':
        plt.hist(q_muhat_MLL_mu, 25, weights=weights, histtype='step', color='green', linewidth=2, label=r'$\hat{\mu}=1$')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("$q$",fontsize=16)
    plt.ylabel("Fraction of pseudo experiments",fontsize=16)
    plt.grid()
    plt.legend(fontsize=14)
    plt.show()
    
    
    
    # With the calculation of mu_hat
    muhat_mean_MLL = np.mean(muhat_selected_MLL_list)

    Z_MLL = abs( np.median(q_muhat_MLL) )**0.5
    Z_MLL_std = np.std(q_muhat_MLL) / (2.*Z_MLL)
    
    
    # With the approximation of mu_hat
    Z_MLL_mu = abs( np.median(q_muhat_MLL_mu) )**0.5
    Z_MLL_std_mu = np.std(q_muhat_MLL_mu) / (2.*Z_MLL_mu)
    
    
    return Z_MLL, Z_MLL_std, muhat_mean_MLL, Z_MLL_mu, Z_MLL_std_mu

