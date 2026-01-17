import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .plot_utils import plot_weight_tiuta_sort

def indept_sample_array(probability):
    sampled_indices = []
    while len(sampled_indices) == 0:
        sampled_indices = np.where(np.random.rand(len(probability)) < probability)[0]
    return sampled_indices



def buildMP_indept(X,Y,n_ratio,m_ratio,prob_I=None,prob_F=None,delta=1):
    N = len(X)
    M = len(X[0])
    n = int(n_ratio * N)
    m = int(m_ratio * M)
    r = np.random.RandomState()
    ## index of minipatch
    if prob_I is None:
        idx_I = np.sort(r.choice(N, size=n, replace=False))
    else:
        idx_I = np.sort(r.choice(N, size=n, replace=False, p=prob_I))

    if prob_F is None:
        idx_F = indept_sample_array(np.ones(M)*m_ratio)
    else:
        idx_F = indept_sample_array(prob_F)

    ## record which obs/features are subsampled
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]

def predictMP_indept(X,Y,X1, n_ratio,m_ratio,B,fit_func,prob_I=None,prob_F=None,delta=1):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B):
        [idx_I,idx_F,x_mp,y_mp] = buildMP_indept(X,Y,n_ratio,m_ratio,prob_I,prob_F,delta)
        predictions.append(fit_func(x_mp,y_mp,X1[:, idx_F]))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True
    return [np.array(predictions),in_mp_obs,in_mp_feature]

def MPRegFeatureScore_indept(X,Y,X1,Y1,n_ratio,m_ratio,K,fit_func,prob_I,prob_F,delta):
    N=len(X)
    M = len(X[0])
    [predictions,in_mp_obs,in_mp_feature]= predictMP_indept(X,Y,np.vstack((X,X1)),n_ratio,m_ratio,K,fit_func,prob_I=prob_I,prob_F=prob_F,delta=delta)

    predictions_train = predictions[:,:N]
    predictions_test = predictions[:,N:]

    mse_train = np.mean((Y-predictions_train.mean(0))**2)
    mse_test = np.mean((Y1-predictions_test.mean(0))**2)

    #############################
    ## Find LOO
    ##############_train##############
    ######## b_keep gives length N Series; the ith item includes the MP indices that exclude sample i
    b_keep = pd.DataFrame(~in_mp_obs).apply(lambda i: np.array(i[i].index))
    ####### absolute error function; compute LOO
    # resids_LOO = list(map(lambda i: np.abs(Y[i] - predictions_train[b_keep[i],i].mean()),range(N)))
    resids_LOO = list(map(lambda i: (np.abs(Y[i] - predictions_train[b_keep[i],i].mean()))**2,range(N)))

    LOO_sd = np.array(list(map(lambda i: predictions_train[b_keep[i],i].std(),range(N)))).mean()
    LOO_mean = np.array(list(map(lambda i: predictions_train[b_keep[i],i].mean(),range(N)))).mean()

    ################################
    ######## FIND LOCO_LOO of
    #############################
    
    results = Parallel(n_jobs=-1)(delayed(get_loco)(i,j,in_mp_feature,in_mp_obs,predictions_train) for i in range(N) for j in range(M))
    ####### ress includes all the LOCO_LOO residuals
    ress = pd.DataFrame(results)
    ress['i'] = np.repeat(range(N),M)
    ress['j'] = np.tile(range(M),N)
    ress['true_y'] = np.repeat(Y,M)

    ress['resid_loco'] =(np.abs(ress['true_y'] - ress[0]))**2
    ress['resid_loo'] = np.repeat(resids_LOO,M)
    ress['zz'] = ress['resid_loco'] -ress['resid_loo']


    Delta = np.zeros((M,));
    for j in range(M):
        Delta[j] = ress[ress.j==j].zz.mean()


    ###########################
    res_tmp = {}
    res_tmp['Delta']=Delta
    res_tmp['mse_train']=mse_train
    res_tmp['mse_test']=mse_test
    res_tmp['loo'] = np.array(resids_LOO).mean()
    res_tmp['loo_std'] = LOO_sd
    res_tmp['loo_mean'] = LOO_mean
    res_tmp['m']=sum(in_mp_feature[0]==1)
    res_tmp['last_pred'] = predictions[:, -1]
    return res_tmp


def indept_weight_sample_epochtuned(X,Y,X1,Y1,n_ratio,m_ratio,K,fit_func,delta,max_iter,plot=True):
    M = len(X[0])
    kk = 0; prob_I = None; prob_F = None
    res = {}
    if len(K) == 1:
        K_list = np.repeat(K, max_iter)
    else:
        K_list = K


    while kk < max_iter:
        res[kk]=MPRegFeatureScore_indept(X,Y,X1,Y1,n_ratio,m_ratio,K_list[kk],fit_func,prob_I,prob_F,delta)
        if kk > 0:
           if res[kk]["loo"] >= res[kk-1]["loo"]:
              res.popitem()
              break
        ###########################
        ####### Update sampling probability
        ###########################
        weight_tiuta = res[kk]['Delta'] - np.min(res[kk]['Delta']) + 0.001/M

        weight_sort = np.sort(weight_tiuta)[::-1]

        total_sum = sum(weight_sort)
        running_sum = 0
        target_index = 0;
        m = M*m_ratio
        for i in range(M):
          running_sum += weight_sort[i]
          if total_sum - running_sum > (m/delta - i - 1) * weight_sort[i+1]:
            target_index = i
            break
        delta_bar = sum(weight_sort[target_index+1:])/(m/delta-(target_index+1))

        if plot:
            plot_weight_tiuta_sort(weight_sort, target_index, delta_bar, kk, first_k=20)

        weight = np.array([min(delta_tiuta, delta_bar) for delta_tiuta in weight_tiuta])
        sum_weight = sum(weight)
        prob_F = m*weight/sum_weight

        # add sampling probability to res
        res[kk]['prob_F'] = prob_F
        res[kk]['weight'] = weight

        kk = kk + 1

    return res


def get_loco(i,j,in_mp_feature,in_mp_obs,predictions_train):
    ####### b_keep_f is the set of minipatch indices excluding sample i and feature j
    b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
    return predictions_train[b_keep_f,i].mean()