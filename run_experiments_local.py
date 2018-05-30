#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import schirm_tools as schirm
import experiments as expe

data_path        = './data/'
result_path      = './results/'
result_path_comb = './results/'

compute = False

Mall      = np.array([2,3,4,5,6,7,8,12,16,20])
Mact_all  = np.array([1,1,2,2,3,3,4, 4, 4, 4])
n_tests   = 100
n_jobs    = 1

if compute:
    ''' in practice, it is convenient to compute these loops in parallel on a cluster '''
    for i in range(len(Mall)):
        # simulated data
        run_name = ''
        for job_id in range(1,n_jobs+1):
            expe.run_testing_with_simulated_data(Mall[i],Mact_all[i],n_tests,job_id,run_name,data_path,'GSM1599500_K562_cells',result_path)    
        expe.combine_results(Mall[i],Mact_all[i],n_jobs,run_name,result_path,result_path_comb)
        # experimental data
        run_name = 'real'
        for job_id in range(1,n_jobs+1):
            expe.run_testing_with_real_data(Mall[i],Mact_all[i],n_tests,job_id,run_name,data_path,'GSM1599500_K562_cells_zeros_removed',result_path)
        expe.combine_results(Mall[i],Mact_all[i],n_jobs,run_name,result_path,result_path_comb)

#run_name = ''
#for i in range(len(Mall)):
#    M = Mall[i]
#    Mact_temp = Mact_all[i]
#    df1 = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p1' + '.csv',index_col=0)
#    df2 = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p2' + '.csv',index_col=0)
#    rhat    = np.loadtxt(result_path_comb+run_name+'rhat_'   + str(M) + '_' + str(Mact_temp) + '.txt')
#    
#    name = run_name +'res_'    + str(M) + '_' + str(Mact_temp)
#    
#    schirm.roc_auc_comparison(df1,fig_name=name + 'rorauc_comp')
#    schirm.roc_and_score_dists(df1,fig_name=name + 'ror_and_score_dists')
#    schirm.param_estimates_vs_true_values(df1,df2,fig_name=name+'est_vs_true')
#    schirm.illust_convergence_diag(rhat,fig_name=name+'rhat')
#
#run_name = 'real'
#for i in range(len(Mall)):
#    M = Mall[i]
#    Mact_temp = Mact_all[i]
#    df1 = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p1' + '.csv',index_col=0)
#    df2 = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p2' + '.csv',index_col=0)
#    rhat    = np.loadtxt(result_path_comb+run_name+'rhat_'   + str(M) + '_' + str(Mact_temp) + '.txt')
#    
#    name = run_name +'res_'    + str(M) + '_' + str(Mact_temp)
#    
#    schirm.roc_auc_comparison(df1,fig_name=name + 'rorauc_comp')
#    schirm.roc_and_score_dists(df1,fig_name=name + 'ror_and_score_dists')
#    schirm.illust_convergence_diag(rhat,fig_name=name+'rhat')
    
#expe.illust_AUC_summary(Mall,Mact_all,'','real',result_path_comb)
#expe.illust_errors(Mall,Mact_all,'',result_path_comb)
expe.param_est_fig('',result_path_comb)