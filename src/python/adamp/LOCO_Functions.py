### Notice: "# code: ***" code starts from "#"

### Libraries #####################################################################################################

import os
import numpy as np
import pandas as pd
import pickle

from scipy.stats import *
from random import sample
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

### Set Parameters ################################################################################################

### Get m, n, Kb and beta
def get_mn(M,N,a):
    n = int(N**a)
    m = int(min(max(0.1*M, 15),0.5*n))
    return m,n

def get_mn1(M,N):
    n = int(0.5*N)
    m = int(min(0.3*M,0.5*n))
    return m,n

def get_Kb(M,N,m,n,k1,k2,k3,k4,k5):
    c = M/m * N/n
    Kb = [int(k1* c), int(k2* c), int(k3* c), int(k4* c), int(k5* c)]
    return Kb

def get_beta1(beta1, beta2):
    betas = []
    for i in beta1:
        betas.append([i]+beta2)
    return betas

def get_beta5(beta1, beta2):
    betas = []
    for i in beta1:
        betas.append(beta2+[i])
    return betas

### Inference #####################################################################################################

def ztest(z,alpha,MM=1,bonf_correct=False):
    try:
        s = np.std(z)
    except:
        return [0,0,0,0]
    l = len(z)
    s = np.std(z)
    if s==0:
        return [0,0,0,0]
    m = np.mean(z)
    pval1 = 1-norm.cdf(m/s*np.sqrt(l))

    pval2 = 2*(1-norm.cdf(np.abs(m/s*np.sqrt(l))))

    # Apply Bonferroni correction for M tests
    if bonf_correct:
        pval1= min(MM*pval1,1)
        pval2= min(MM*pval2,1)
        alpha = alpha/MM
        
    q = norm.ppf(1-alpha/2)
    left  = m - q*s/np.sqrt(l)
    right = m + q*s/np.sqrt(l)

    return [pval1,pval2, left,right]

### Method: LOCO-AdaMP ############################################################################################

def indep_sampling_probability(z, m, indep_delta):    
    #z = errorLOCO - errorLOO
    prob = np.zeros(len(z))
    for i in range(len(z)):
        prob[i] = z[i].mean()
    prob = prob - min(prob) 

    prob_sort = np.sort(prob)[::-1]
    
    total_sum = sum(prob_sort)
    running_sum = 0
    target_index = 0
    for i in range(len(prob_sort)):
        running_sum += prob_sort[i]
        if total_sum - running_sum > (m/indep_delta - i - 1) * prob_sort[i+1]:
            target_index = i
            break

    prob_bar = sum(prob_sort[target_index+1:])/(m/indep_delta-(target_index+1))
    weight = np.array([min(prob_j, prob_bar) for prob_j in prob])
    indep_samp_prob = m*weight/sum(weight)
    return indep_samp_prob

def indept_sample_array(probability):
    sampled_indices = []
   
    for i in range(len(probability)):
        rn = np.random.rand()
        if  rn < probability[i]: # Probability of including element i
            sampled_indices.append(i)
    # code:sampled_arr = probability[sampled_indices]
    return sampled_indices

def get_Q(prob, F):
    if len(F) == 0:
        Q_F = 0
    else:
        F1 = np.setdiff1d(np.arange(len(prob)),F)
        Q_F = np.prod(prob[F])*np.prod(1-prob[F1])/np.prod(1-prob)
    return Q_F

def indep_buildMP(X,Y,n,m, indep_samp_prob, indep = False):
    N = len(X)
    M = len(X[0])
    
    # index of minipatch
    idx_I = np.sort(np.random.choice(N, size=n, replace=False))
    idx_F = np.array([])# uniform sampling of subset of observations
    if indep:
        #i = 0
        while len(idx_F) == 0:
            # code: i = i+1
            idx_F = np.sort(indept_sample_array(indep_samp_prob))
    else:
        idx_F = np.sort(np.random.choice(M, size=m, replace=False, p=indep_samp_prob)) 
    # record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]

def indep_predictMP(X,Y,X1, n, m,B,fit_funct,paras,indep_samp_prob, indep = False):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B): 
        [idx_I,idx_F,x_mp,y_mp] = indep_buildMP(X,Y,n, m, indep_samp_prob, indep = indep)
        predictions.append(fit_funct(x_mp,y_mp,X1[:, idx_F],paras))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]

def LOCOAdaMPReg(X,Y,X1, Y1, n, m,B,fit_funct,paras,indep_samp_prob, find_kb = False, indep = False, selected_features = [],n_features = 10,alpha=0.1, bonf=False):

    N=len(X)
    M = len(X[0])
    N1=len(X1)

    [predictions,in_mp_obs,in_mp_feature]= indep_predictMP(X,Y,np.vstack((X,X1)),n ,m,B,fit_funct,paras, indep_samp_prob, indep = indep)
    predictions_train = predictions[:,:N]
    predictions_test = predictions[:,N:]
      
    # Re-fit after dropping each feature
    resids_LOO,resids_LOCO = np.zeros(N),np.zeros(N)
    resids_LOO_sq,resids_LOCO_sq = np.zeros(N),np.zeros(N)
    y_new =np.zeros((N,N1), dtype=float)
    zeros=False

    if find_kb:
        gamma1 = []
        gamma2 = []

        I_kb = np.random.choice(N, size=50, replace=True)
        for i in I_kb:
            b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            #print(b_keep)
            sel_2 = sample(b_keep,2)
            gamma1.append(np.square(predictions_train[sel_2[0],i] - predictions_train[sel_2[1],i]))
            gamma2.append(np.mean(predictions_train[b_keep,i]))
        
        print("K:",B, ", Valid:", np.sqrt(np.mean(gamma1)/B) <= abs(np.mean(gamma2)/10))

    ### Find LOO
    for i in range(N):
        ## find MP has no i but has j
        b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
                  
        if len(b_keep)>0:
            resids_LOO[i]= np.abs(Y[i] - predictions_train[b_keep,i].mean())
            resids_LOO_sq[i]= np.square(Y[i] - predictions_train[b_keep,i].mean())
    
    ### FIND LOCO
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z,inf_z_sq = np.zeros((len(ff),4)), np.zeros((len(ff),4))
    z,z_sq={},{}
    resids_LOCOs ={}
    # code: resids_LOCOs_sq={}

    for idd,j in enumerate(ff):
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            resids_LOCO[i]= np.abs(Y[i] - predictions_train[b_keep_f,i].mean())
            # code: resids_LOCO_sq[i]= np.square(Y[i] - predictions_train[b_keep_f,i].mean())
            
        zz = resids_LOCO - resids_LOO

        z[idd] = zz[~np.isnan(zz)]
        resids_LOCOs[idd] = resids_LOCO.copy()
        # code: zz = resids_LOCO_sq - resids_LOO_sq
        # code: z_sq[idd] = zz[~np.isnan(zz)]
        # code: resids_LOCOs_sq[idd] = resids_LOCO_sq.copy()
                    
        if len(z)==0:
            inf_z[idd]= [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM = len(ff),bonf_correct =bonf)
            # code: inf_z_sq[idd] = ztest(z_sq[idd],alpha,MM = len(ff),bonf_correct =bonf)

    uhat_test=predictions_test.mean(0)
    #mps1 = list(set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
    #mps2 = list(set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
    #uhat1_test=predictions_test[mps1,:].mean(0)
    #uhat2_test=predictions_test[mps2,:].mean(0)

    var,target,err1,err2,err2_sq,err2_,err2_sq_ = {},{},{},{},{},{},{}
    for idd,j in enumerate(ff): ## ff include feature of interest
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)))
        uhat_j_test= predictions_test[b_keep_f,:].mean(0) ## TEST ERROR WITHOUT J
        
        err1[idd] = np.abs(Y1-uhat_j_test)
        err2[idd] = np.abs(Y1-uhat_test)
        target[idd] = np.mean(err1[idd]-err2[idd])
        var[idd]= np.std(err1[idd]-err2[idd])
    
        err2_sq[idd] = (Y1-uhat_test)**2
    
    # code: inf_z_ = np.zeros((n_features,4))
    for idd in range(n_features):
        err2_[idd] = err2[idd]
        err2_sq_[idd] = err2_sq[idd]
        # code: inf_z_[idd] =inf_z[idd]
        
    ### result
    res= {}
    res['err2_'] = err2_
    res['err2_sq_'] = err2_sq_
    res['resids_LOO']=resids_LOO
    res['resids_LOO_sq']=resids_LOO_sq
    res['inf']=inf_z
    res['indep_samp_prob'] = indep_samp_prob
    res['z']=z
    res['target']=target
    res['variance']=var 
    return res

### Method: LOCO-MP ###############################################################################################

def buildMP(X,Y,n,m):
    N = len(X)
    M = len(X[0])
    ## index of minipatch
    idx_I = np.sort(np.random.choice(N, size=n, replace=False)) # uniform sampling of subset of observations
    idx_F = np.sort(np.random.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]

def predictMP(X,Y,X1,n,m,K_locomp,fit_funct):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    in_mp_obs,in_mp_feature = np.zeros((K_locomp,N),dtype=bool),np.zeros((K_locomp,M),dtype=bool)
    predictions=[]
    for b in range(K_locomp):        
        [idx_I,idx_F,x_mp,y_mp] = buildMP(X,Y,n,m)
        predictions.append(fit_funct(x_mp,y_mp,X1[:, idx_F]))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]

def LOCOMPReg(X,Y,X1, Y1, n,m,K_locomp,fit_funct,selected_features=[],n_features = 10,alpha=0.1,bonf=False):

    N=len(X)
    M = len(X[0])
    N1=len(X1)

    [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,np.vstack((X,X1)),n,m,K_locomp,fit_funct)
    predictions_train = predictions[:,:N]
    predictions_test = predictions[:,N:]
    
    # Re-fit after dropping each feature
    resids_LOO,resids_LOCO = np.zeros(N),np.zeros(N)
    resids_LOO_sq,resids_LOCO_sq = np.zeros(N),np.zeros(N)
    y_new =np.zeros((N,N1), dtype=float)
    zeros=False

    ### Find LOO
    for i in range(N):
        # find MP has no i but has j
        b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
                  
        if len(b_keep)>0:
            resids_LOO[i]= np.abs(Y[i] - predictions_train[b_keep,i].mean())
            resids_LOO_sq[i]= np.square(Y[i] - predictions_train[b_keep,i].mean())

    ### FIND LOCO
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z,inf_z_sq = np.zeros((len(ff),4)), np.zeros((len(ff),4))
    z,z_sq={},{}
    resids_LOCOs = {}
    resids_LOCOs_sq={}
    # code: resids_LOCOs_sq={}

    for idd,j in enumerate(ff):
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            resids_LOCO[i]= np.abs(Y[i] - predictions_train[b_keep_f,i].mean())
            # code: resids_LOCO_sq[i]= np.square(Y[i] - predictions_train[b_keep_f,i].mean())
            resids_LOCO_sq[i]= np.square(Y[i] - predictions_train[b_keep_f,i].mean())
        zz = resids_LOCO - resids_LOO
        z[idd] = zz[~np.isnan(zz)]
        resids_LOCOs[idd] = resids_LOCO.copy()

        # code: zz = resids_LOCO_sq - resids_LOO_sq
        zz = resids_LOCO_sq - resids_LOO_sq
        # code: z_sq[idd] = zz[~np.isnan(zz)]
        # code: resids_LOCOs_sq[idd] = resids_LOCO_sq.copy()
        zz_sq = resids_LOCO_sq - resids_LOO_sq
        z_sq[idd] = zz_sq[~np.isnan(zz_sq)]
        resids_LOCOs_sq[idd] = resids_LOCO_sq.copy()
               
        if len(z)==0:
            inf_z[idd]= [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =bonf)
            # code: inf_z_sq[idd] = ztest(z_sq[idd],alpha,MM=len(ff),bonf_correct =bonf)

    uhat_test=predictions_test.mean(0)
    mps1 = list(set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
    mps2 = list(set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
    uhat1_test=predictions_test[mps1,:].mean(0)
    uhat2_test=predictions_test[mps2,:].mean(0)
    
    var,target,err1,err2,err2_sq,err2_,err2_sq_ = {},{},{},{},{},{},{}
    for idd,j in enumerate(ff): # ff include feature of interest
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)))
        uhat_j_test= predictions_test[b_keep_f,:].mean(0) # TEST ERROR WITHOUT J
        
        err1[idd] = np.abs(Y1-uhat_j_test)
        err2[idd] = np.abs(Y1-uhat_test)
        target[idd] = np.mean(err1[idd]-err2[idd])
        var[idd]= np.std(err1[idd]-err2[idd])
        
        err2_sq[idd] = (Y1-uhat_test)**2
        
    # code: inf_z_ = np.zeros((n_features,4))
    Delta = np.full(M, np.nan)
    for idd, j in enumerate(ff):
        # squared LOCO minus squared LOO for all i, then average
        diff_sq = resids_LOCOs_sq[idd] - resids_LOO_sq
        Delta[j] = np.nanmean(diff_sq)

    for idd in range(n_features):
        err2_[idd] = err2[idd]
        err2_sq_[idd] = err2_sq[idd]
        # code: inf_z_[idd] =inf_z[idd]
        
    ### Result
    
    res= {}
    res['err2_'] = err2_
    res['err2_sq_'] = err2_sq_
    res['resids_LOO']=resids_LOO
    res['resids_LOO_sq']=resids_LOO_sq
    res['inf']=inf_z
    res['z']=z
    res['target']=target 
    res['variance']=var

    res['Delta'] = Delta
   
    return res

### Method: LOCO-Split ############################################################################################

# def LOCOSplitReg(X,Y,X1,Y1,fit_funct,ratio,paras,selected_features=[],n_features = 10,alpha=0.1,bonf=False):
#     N=len(X)
#     M = len(X[0])
#     N1=len(X1)
    
#     # uniform sampling of subset of observations
#     idx_I = np.sort(np.random.choice(N, size=int(ratio*N), replace=False))
#     x_train=X[idx_I,:]
#     y_train=Y[np.ix_(idx_I)]
#     out_I =list(set(range(N))-set(idx_I))
#     x_val = X[out_I]
#     y_val = Y[out_I]
    
#     #if fit_func == RidgeReg:
#     #    fit_func2 = RidgeCVReg
#     #    prediction = fit_func2(x_train,y_train,np.vstack((x_val,X1)),paras)
#     #if fit_func == DecisionTreeReg:
#     #    fit_func2 = RFreg
#     #    prediction = fit_func2(x_train,y_train,np.vstack((x_val,X1)),paras)
#     #if fit_func == KernelRidgeReg:
#     #    fit_func2 = KernelRidgeReg2
#     #    prediction = fit_func2(x_train,y_train,np.vstack((x_val,X1)),paras)
    
#     prediction = fit_funct(x_train,y_train,np.vstack((x_val,X1)))
#     predictions  = prediction[:len(y_val)]
#     predictions_test = prediction[len(y_val):]

#     resids_split = np.abs(y_val - predictions)
#     resids_split_sq = (y_val - predictions)**2

#     resids_split_test = np.abs(Y1 - predictions_test)
#     # code: resids_split_test_sq = np.square(Y1 - predictions_test)

#     # Re-fit after dropping each feature    
#     if len(selected_features)==0:
#         ff = range(M)
#     else:
#         ff=selected_features
#     inf_z,inf_z_sq= np.zeros((len(ff),4)),np.zeros((len(ff),4))
    
#     uhat_j_test={}
    
#     for idd,j in enumerate(ff):
#         # Train on the first part, without variable j, predict on 2nd without j 
#         out_js = fit_funct(np.delete(x_train,j,1),y_train,np.delete(np.vstack((x_val,X1)),j,1)) 
        
#         out_j  = out_js[:len(y_val)]
#         uhat_j_test[idd] = out_js[len(y_val):]
    
        
#         resids_drop=np.abs(y_val - out_j)
#         resids_drop_sq = np.mean((y_val - out_j)**2)
#         z = resids_drop - resids_split
#         # code: z_sq= resids_drop_sq - resids_split_sq
#         inf_z[idd] = ztest(z,alpha,bonf_correct =bonf)
#         # code: inf_z_sq[idd] = ztest(z_sq,alpha,bonf_correct =bonf)

        
#     uhat_test=predictions_test ## TEST ERROR WITH J 
#     target,err1,err2,err2_sq,err2_,err2_sq_ = {},{},{},{},{},{}
#     for idd,j in enumerate(ff): # ff include feature of interest
#         err1[idd] = np.abs(Y1-uhat_j_test[idd])
#         err2[idd] = np.abs(Y1-uhat_test)
#         target[idd] = np.mean(err1[idd]-err2[idd])

#         err2_sq[idd] = (Y1-uhat_test)**2
#     # code: inf_z_ = np.zeros((n_features,4))
#     for idd in range(n_features):
#         err2_[idd] = err2[idd]
#         err2_sq_[idd] = err2_sq[idd]
#         # code: inf_z_[idd] =inf_z[idd]

#     ### Result
#     res= {}
#     res['err2_'] = err2_
#     res['err2_sq_'] = err2_sq_
#     res['resids_LOO']=resids_split
#     res['resids_LOO_sq']=resids_split_sq
#     res['inf']=inf_z
#     res['z']=z
#     res['target']=target 
   
#     return res

def LOCOSplitReg(X,Y,X1,Y1,fit_funct,ratio,selected_features=[],n_features = 10,alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)
    
    # uniform sampling of subset of observations
    idx_I = np.sort(np.random.choice(N, size=int(ratio*N), replace=False))
    x_train = X[idx_I,:]
    y_train = Y[np.ix_(idx_I)]
    out_I = list(set(range(N)) - set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    
    prediction = fit_funct(x_train, y_train, np.vstack((x_val, X1)))
    predictions      = prediction[:len(y_val)]
    predictions_test = prediction[len(y_val):]

    # baseline (with all features) validation residuals
    resids_split    = np.abs(y_val - predictions)
    resids_split_sq = (y_val - predictions)**2

    resids_split_test = np.abs(Y1 - predictions_test)
    # resids_split_test_sq = np.square(Y1 - predictions_test)  # not needed for Delta

    # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = range(M)
    else:
        ff = selected_features

    inf_z, inf_z_sq = np.zeros((len(ff),4)), np.zeros((len(ff),4))
    
    uhat_j_test = {}
    resids_drop_sq_all = {}      # NEW: store per-feature squared residuals on val
    
    for idd, j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd + X1 without j 
        out_js = fit_funct(
            np.delete(x_train, j, 1),
            y_train,
            np.delete(np.vstack((x_val, X1)), j, 1)
        ) 
        
        out_j  = out_js[:len(y_val)]
        uhat_j_test[idd] = out_js[len(y_val):]
    
        # validation residuals without feature j
        resids_drop = np.abs(y_val - out_j)
        resids_drop_sq_vec = (y_val - out_j)**2   # NEW: per-observation sq residuals

        # store for Delta
        resids_drop_sq_all[idd] = resids_drop_sq_vec   # NEW

        # z-test on absolute residuals
        z = resids_drop - resids_split
        # z_sq = resids_drop_sq_vec - resids_split_sq  # optional squared version

        inf_z[idd] = ztest(z, alpha, bonf_correct=bonf)
        # inf_z_sq[idd] = ztest(z_sq, alpha, bonf_correct=bonf)  # optional

    # NEW: feature importance Delta based on validation squared residuals
    Delta = np.full(M, np.nan)
    for idd, j in enumerate(ff):
        diff_sq = resids_drop_sq_all[idd] - resids_split_sq
        Delta[j] = np.nanmean(diff_sq)

    uhat_test = predictions_test  ## TEST ERROR WITH all features

    target, err1, err2, err2_sq, err2_, err2_sq_ = {},{},{},{},{},{}
    for idd, j in enumerate(ff):  # ff include feature of interest
        err1[idd] = np.abs(Y1 - uhat_j_test[idd])
        err2[idd] = np.abs(Y1 - uhat_test)
        target[idd] = np.mean(err1[idd] - err2[idd])

        err2_sq[idd] = (Y1 - uhat_test)**2

    # inf_z_ = np.zeros((n_features,4))
    for idd in range(n_features):
        err2_[idd] = err2[idd]
        err2_sq_[idd] = err2_sq[idd]
        # inf_z_[idd] = inf_z[idd]

    ### Result
    res = {}
    res['err2_'] = err2_
    res['err2_sq_'] = err2_sq_
    res['resids_LOO'] = resids_split
    res['resids_LOO_sq'] = resids_split_sq
    res['inf'] = inf_z
    res['z'] = z
    res['target'] = target 
    res['Delta'] = Delta        # NEW
   
    return res

