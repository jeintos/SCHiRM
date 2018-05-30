#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import schirm_tools as schirm

np.random.seed(100)

data_path      = './data/'
data_file_name = 'GSM1599500_K562_cells' # we use data to simulate realistic data

M    = 6
Mact = 3

# load model and specify priors
stan_model   = schirm.define_model(compile_model = False)
prior_params = {'beta0_prior_std'   :10.0,
                'beta_prior_std'    :1.0,
                'log_mux_prior_mean':-1.64,
                'log_mux_prior_std' :1.84,
                'alpha_prior_std'   :0.15}

# true and est. normalization constants
c_true          = np.loadtxt(data_path+data_file_name+'_norm_consts.txt') # cell sizes are taken form real data
norm_consts_est = np.loadtxt(data_path+'jittered_norm_consts.txt')

# simulate input/output read counts using given parameters
N         = len(c_true) # number of cells
alpha     = 0.3
mux_true  = np.array([2,4,3,2,5,4])
beta0     = -1
beta      = np.array([0.7,0.8,-0.75,0,0,0])
X, log_CX = schirm.sim_indep_data(mux_true,alpha,c_true,N)
y, log_cy = schirm.sim_output_data(log_CX,c_true,alpha,beta0,beta,N)

# run inference
fit                  = schirm.run_inference(X,y,prior_params,norm_consts_est,stan_model)
result_p1, result_p2 = schirm.fit2table(fit,'',M)
psrf                 = fit.summary()['summary'][(M+1)*N:,-1]

true_params = {'beta0': beta0,'beta':beta,'alpha':alpha,'mux_true':mux_true}
schirm.illust_post_summary_A(fit, M,true_params)

plt.show()