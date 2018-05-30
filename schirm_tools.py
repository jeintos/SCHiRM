#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from scipy.stats.stats import pearsonr
from scipy.stats import truncnorm
from scipy import stats
from sklearn.metrics import roc_curve, auc
from matplotlib.gridspec import GridSpec
import seaborn as sns

def define_model(compile_model = False):
    model_name = 'model.pkl'
    if compile_model:
        code= """
            data {
                int<lower=1>  N;          // number of cells
                int<lower=1>  M;          // number of inputs
                int<lower=0>  X[N,M];     // input matrix
                int<lower=0>  y[N];       // output vector
                real<lower=0> beta0_std;  // std for beta0
                real<lower=0> beta_std;   // std for beta
                real          log_mux_mean;
                real<lower=0> log_mux_std;// std for mux
                real<lower=0> alpha_std;  // std for alpha
                vector[N]     log_c;      // logarithmic normalization constants
            }
            parameters {
                matrix[N,M]   log_CX;     // log-parameters for Poisson inputs
                vector[N]     log_Cy;
                vector[M]     log_mux;
                real          beta0;
                vector[M]     beta;
                real<lower=0> alpha;
            }
            model {
                log_mux ~ normal(log_mux_mean, log_mux_std);
                beta0   ~ normal(0, beta0_std);
                beta    ~ normal(0, beta_std);
                alpha   ~ normal(0, alpha_std); // half-normal prior
                // prior for log-parameters of Poisson inputs
                for (j in 1:M){
                    for (i in 1:N){
                        log_CX[i,j] ~ normal(log_mux[j] - 0.5  * log(1 + alpha),
                                             sqrt(log(1 + alpha)));
                    }
                }
                log_Cy ~ normal(log_CX*beta + beta0, sqrt(log(1 + alpha)));
                // likelihood
                for (i in 1:N){
                    y[i] ~ poisson_log(log_Cy[i] + log_c[i]);
                    for (j in 1:M){
                        X[i,j] ~ poisson_log(log_CX[i,j] + log_c[i]);
                    }
                }
            }
            """
        from pystan import StanModel
        # compile
        stan_model = StanModel(model_code=code)
        # save the model to the file
        with open(model_name, 'wb') as f:
            pickle.dump(stan_model, f)
    else:
        stan_model = pickle.load(open(model_name, 'rb'))
    return stan_model

def run_inference(X,y,prior_params,norm_consts,stan_model,N_iter = 500,N_chains  = 4):
    N, M = X.shape
    # inference
    dat = {'N': N,'M':M,'X': X,'y': y,'log_c': np.log(norm_consts),
           'beta0_std': prior_params['beta0_prior_std'],'beta_std': prior_params['beta_prior_std'],
           'log_mux_mean': prior_params['log_mux_prior_mean'],'log_mux_std': prior_params['log_mux_prior_std'],
           'alpha_std': prior_params['alpha_prior_std']}
    with suppress_stdout_stderr():
        fit = stan_model.sampling(data=dat, iter=N_iter, chains=N_chains)
    return fit

def fit2table(fit,name,M):
    result_p1 = pd.DataFrame(index=range(M),data={
            'SCHiRM'+name+'p'  : compute_prob_score(fit['beta']),
            'SCHiRM'+name+'pm' : fit['beta'].mean(axis=0),
            'beta_std'+name  : fit['beta'].std(axis=0),
            'mux_mean'+name  : np.exp(fit['log_mux']).mean(axis=0),
            'mux_std'+name   : np.exp(fit['log_mux']).std(axis=0)})
    result_p2 = pd.DataFrame(index=np.array([0]),data={
            'beta0_mean'+name: fit['beta0'].mean(),
            'beta0_std'+name : fit['beta0'].std(),
            'alpha_mean'+name: fit['alpha'].mean(),
            'alpha_std'+name : fit['alpha'].std() })
    return result_p1, result_p2

def run_standard_reg(X,y,norm_consts):
    N, M = X.shape
    # normalize and log-transform
    logXn = np.log(1+(X/norm_consts[:,None]))
    logyn = np.log(1+(y/norm_consts))
    # OLS
    ols = LinearRegression()
    ols.fit(logXn,logyn)
    # LASSO
    clf = LassoCV()
    clf.fit(logXn,logyn)
    # ELASTIC NET
    regr = ElasticNetCV()
    regr.fit(logXn,logyn)
    # RIDGE REGRESSION
    ridr = RidgeCV()
    ridr.fit(logXn,logyn)
    # correlation
    r = np.zeros(M)
    for i in range(M):
        r[i] = pearsonr(logXn[:,i],logyn)[0]
    # collect
    result_p1 = pd.DataFrame(data={
            'OLS' : ols.coef_,
            'LASSO' : clf.coef_,
            'ENET' : regr.coef_,
            'RIDGE' : ridr.coef_,
            'PCC'     : r})
    result_p2 = pd.DataFrame(index = np.array([0]),data={
            'beta0_ols': ols.intercept_,
            'beta0_las': clf.intercept_,
            'beta0_ene': regr.intercept_,
            'beta0_rid': ridr.intercept_})
    return result_p1, result_p2

def sim_truncnormal_dist(loc,scale,lb,ub,N):
    a, b = (lb - loc) / scale, (ub - loc) / scale
    return truncnorm.rvs(a, b, loc, scale, size=N)

def sim_indep_data(mu,alpha,norm_consts,N):
    # mu is an M-vector containing the means of the log-normal distributions
    # N is the number of cells
    a      = np.log(mu) - 0.5*np.log(1 + alpha) # mean in the log-scale
    b      = np.log(1 + alpha)                  # variance in the log-scale
    log_CX = np.random.multivariate_normal(a, b*np.eye(len(mu)), size=N)
    X      = np.random.poisson(np.exp(log_CX)*norm_consts[:,None])
    return X, log_CX

def sim_beta(M,bounds,Mact):
    beta0 = np.random.uniform(bounds['beta0'][0],bounds['beta0'][1],1)
    beta  = np.zeros(M)
    if Mact == None:
        if M%2 == 0:
            Mact = int(M/2)
        else:
            Mact = int(np.ceil(M/2)) - np.random.randint(2)
    # simulate values for beta and the corresponding output
    beta[0:Mact] = np.random.uniform(bounds['beta'][0],bounds['beta'][1],Mact)
    return beta0, beta

def sim_output_data(log_CX,norm_consts,alpha,beta0,beta,N):
    log_Cy = beta0 + np.dot(log_CX,beta) + np.random.normal(0, np.sqrt(np.log(1+alpha)), N)
    y      = np.random.poisson(np.exp(log_Cy)*norm_consts)
    return y, log_Cy

def compute_prob_score(samples):
    N_samp = samples.shape[0]
    pos_prob = np.sum(samples > 0,axis=0)/N_samp
    neg_prob = np.sum(samples < 0,axis=0)/N_samp
    return np.maximum(pos_prob,neg_prob)

def estimate_normconsts(X):
    tot_exp = np.sum(X,axis=1)
    return len(tot_exp)*tot_exp/np.sum(tot_exp)

def compute_norm_constants_and_mean_expression_prior(df,lb,ub,illust = False):    
    # estimate normalization constants
    c      = estimate_normconsts(df.as_matrix())
    c_mean = np.round(np.mean(np.log(c)),2)
    c_std  = np.round(np.std(np.log(c)),2)
    
    # remove the genes which have zero expression in all cells
    df = df.iloc[:,df.sum().as_matrix()>0]
    
    # compute average expression of each gene
    m = df.mean().as_matrix()
    
    # fit a normal distribution to the logarithmic means (full data, only zeros removed)
    m_mean = np.round(np.mean(np.log(m)),2)
    m_std  = np.round(np.std(np.log(m)),2)
    
    # visualize
    if illust:
        plt.subplot(2,2,1)
        plt.hist(c,density=True)
        plt.xlabel('Norm. constant')
    
        plt.subplot(2,2,2)
        plt.hist(np.log(c),density=True)
        x   = np.linspace(np.min(np.log(c)), -np.min(np.log(c)), 100)
        pdf = stats.norm.pdf(x,loc=c_mean,scale=c_std)
        plt.plot(x, pdf, label="PDF")
        plt.xlabel('log(Norm. constant)')
        plt.title('mean = '+str(c_mean)+', std = '+str(c_std))
            
        plt.subplot(2,2,3)
        plt.hist(np.log(m),density=True)
        x   = np.linspace(-np.max(np.log(m)), np.max(np.log(m)), 100)
        pdf = stats.norm.pdf(x,loc=m_mean,scale=m_std)
        plt.plot(x, pdf, label="PDF")
        plt.plot([lb,lb],[0,np.max(pdf)], '--k')
        plt.plot([ub,ub],[0,np.max(pdf)], '--k')
        plt.xlabel('log(Aver. expression)')
        plt.title('mean = '+str(m_mean)+', std = '+str(m_std))
    return df, c, m_mean, m_std

def compute_jittered_norm_constants(m_mean,m_std,c,N,alpha,Mtot = 5000):
    # simulate data to generate realistically jittered estimates of norm. constants
    D, log_CD = sim_indep_data(np.exp(np.random.normal(m_mean,m_std,size=Mtot)),alpha,c,N)
    # estimate normalization constants from sim. data
    return estimate_normconsts(D)

def illust_convergence_diag(rhat,fig_name=None):
    fig = plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.hist(rhat.flatten())
    plt.xlabel('Rhat')
    plt.subplot(1,2,2)
    plt.imshow(rhat,cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('Rhat')
    fig.tight_layout()
    if fig_name != None:
        fig.savefig('./figs/'+fig_name+'.pdf', format='pdf', dpi=1200)

def roc_auc_comparison(df,names = ['SCHiRMp','SCHiRMpm',
                                   'OLS', 'LASSO','ENET','RIDGE','PCC'],fig_name=None):
    th = 0
    y_true = np.abs(df['beta_true'].as_matrix()) > th
    
    # compute and plor ROC/AUC
    AUC = np.array([])
    for name in names:
        score = np.abs(df[name].as_matrix())
        temp  = roc_analysis(y_true, score, name, illust = True)
        AUC   = np.append(AUC,temp)
    plt.legend()
    
    # plot AUCs
    fig, ax = plt.subplots()
    y_pos = np.arange(len(names))
    ax.barh(y_pos, AUC, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('AUC')
    if fig_name != None:
        fig.savefig('./figs/'+fig_name+'.pdf', format='pdf', dpi=1200)

def roc_and_score_dists(df,names = ['SCHiRMp','SCHiRMpm',
                                   'OLS', 'LASSO','ENET','RIDGE','PCC'],fig_name=None):
    th = 0
    y_true = np.abs(df['beta_true'].as_matrix()) > th
    
    n_rows = 2
    fig = plt.figure(figsize=(2*len(names),4))
    # compute and plot ROC curves
    for i, name in enumerate(names):
        # roc curves
        plt.subplot(n_rows,len(names),i+1)
        score = np.abs(df[name].as_matrix())
        roc_analysis(y_true, score, name, illust = True, showxy_label = (i==0))
        plt.title(name)
        # pos./neg. score distributions
        plt.subplot(n_rows,len(names),i+1+len(names))
        ind     = np.abs(y_true) > th
        ind_neq = np.abs(y_true) <= th
        plt.hist(score[ind],label='Pos.')
        plt.hist(score[ind_neq],alpha=0.5,label='Neg.')
        if i == 0:
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.legend()
            
    fig.tight_layout()
    # if figure name is given, save the image in .svg format
    if fig_name != None:
        fig.savefig('./figs/'+fig_name+'.pdf', format='pdf', dpi=1200)

def param_estimates_vs_true_values(df1,df2,
    names1 = ['SCHiRMpm', 'OLS', 'LASSO','ENET','RIDGE'],
    names2 = ['beta0_mean','beta0_ols', 'beta0_las', 'beta0_ene', 'beta0_rid'],fig_name=None):
    n_rows = 4
    fig = plt.figure(figsize=(2*len(names1),8))
    # true betas vs est
    for i, name in enumerate(names1):
        plt.subplot(n_rows,len(names1),i+1)
        plt.plot([-1,1],[-1,1],'k')
        plt.scatter(df1['beta_true'].as_matrix(),df1[name].as_matrix(),s = 0.2)
        plt.title(name)
        if i == 0:
            plt.xlabel('True $\\beta$')
            plt.ylabel('Est. $\\beta$')
    # true beta0 vs est
    for i, name in enumerate(names2):
        plt.subplot(n_rows,len(names1),i+1+len(names1))
        plt.plot([-1,1],[-1,1],'k')
        plt.scatter(df2['beta0_true'].as_matrix(),df2[name].as_matrix(),s = 0.2)
        if i == 0:
            plt.xlabel('True $\\beta_0$')
            plt.ylabel('Est. $\\beta_0$')
    
    # true mu vs est
    plt.subplot(n_rows,len(names1),1+2*len(names1))
    plt.plot([0,100],[0,100],'k')
    plt.scatter(df1['mux_true'].as_matrix(),df1['mux_mean'].as_matrix(),s = 0.6)
    plt.xlabel('True $\mu$')
    plt.ylabel('Est. $\mu$')

    # alpha
    plt.subplot(n_rows,len(names1),1+3*len(names1))
    plt.hist(df2['alpha_mean'].as_matrix())
    plt.xlabel('$\\alpha$')
    plt.ylabel('Freq.')

    fig.tight_layout()
    if fig_name != None:
        fig.savefig('./figs/'+fig_name+'.pdf', format='pdf', dpi=1200)

def roc_analysis(y_true, score, lab, illust = False, showxy_label = True):
    fpr, tpr, th = roc_curve(y_true, score, pos_label=True)
    AUC          = auc(fpr, tpr)
    if illust:
        plt.plot(fpr, tpr,label = lab)
        if showxy_label:
            plt.xlabel('FPR')
            plt.ylabel('TPR')
    return AUC

def illust_posterior_violin(S,var_name):
    M = S.shape[1]
    names = []
    for i in range(1,M+1):
        var_name_ind = '$\\' + var_name + '_' + str(i) + '$'
        names.append(var_name_ind)
    df = pd.DataFrame(S,columns=names)
    df = df.melt(var_name='Parameter', value_name='Value')
    #plt.figure()
    sns.set_style("whitegrid")
    sns.violinplot(x="Parameter", y="Value", data=df)

def illust_posterior_kde_hist(s,var_name):
    sns.distplot(s)
    plt.xlabel(var_name)
    plt.ylabel('Density')

def illust_post_summary_A(fit, M, true_params=None):
    fig = plt.figure(figsize=(6,6))
    MS        = 12
    thickness = 2   # horizontal line thickness
    ccc       = 'r' # horizontal line color
    
    gs = GridSpec(4, 2)    
    plt.subplot(gs[2,0])
    illust_posterior_kde_hist(fit['beta0'],'$\\beta_0$')
    if true_params != None:
        beta0 = true_params['beta0']
        plt.plot(beta0,0,'|', mew=thickness, ms=MS, color=ccc)
        #plt.plot([beta0,beta0],[0,0.65],color=ccc)
    
    plt.subplot(gs[2,1])
    illust_posterior_kde_hist(fit['alpha'],'$\\alpha$')
    if true_params != None:
        alpha = true_params['alpha']
        plt.plot(alpha,0,'|', mew=thickness, ms=MS, color=ccc)
        #plt.plot([alpha,alpha],[0,15],color=ccc)
    
    plt.subplot(gs[0:2,0])
    illust_posterior_violin(fit['beta'],'beta')
    if true_params != None:
        beta = true_params['beta']
        plt.plot(beta,'_', mew=thickness, ms=MS, color=ccc)
    
    plt.subplot(gs[0:2,1])
    illust_posterior_violin(np.exp(fit['log_mux']),'mu')
    if true_params != None:
        mux_true = true_params['mux_true']
        plt.plot(mux_true,'_', mew=thickness, ms=MS, color=ccc)
    
    plt.subplot(gs[3,0])
    score = np.abs(fit['beta'].mean(axis=0))
    plt.stem(range(1,M+1),score)
    plt.xticks(range(1,M+1))
    plt.xlabel('Input index')
    plt.ylabel('Post. mean score')
    
    plt.subplot(gs[3,1])
    score = compute_prob_score(fit['beta'])
    plt.stem(range(1,M+1),(score-0.5)/0.5)
    plt.xticks(range(1,M+1))
    plt.xlabel('Input index')
    plt.ylabel('2 x Prob. score - 1')
    
    fig.tight_layout()

# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
import os
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
