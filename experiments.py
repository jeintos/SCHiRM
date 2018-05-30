#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import schirm_tools as schirm
import matplotlib.pyplot as plt

def run_testing_with_simulated_data(M,Mact_temp,n_tests,job_id,run_name,
                                    data_path,data_file_name,result_path):
    if Mact_temp == 0:
        Mact = None
    else:
        Mact = Mact_temp
    
    # load model and determine prior parameters
    stan_model   = schirm.define_model(compile_model = False)
    prior_params = {'beta0_prior_std'   :10.0,
                    'beta_prior_std'    :1.0,
                    'log_mux_prior_mean':-1.64,
                    'log_mux_prior_std' :1.84,
                    'alpha_prior_std'   :0.15}
    
    # simulation setup
    alpha  = 0.3
    c_true = np.loadtxt(data_path+data_file_name+'_norm_consts.txt') # cell sizes are taken form real data
    N      = len(c_true)                                             # number of cells
    bounds = {'beta0':np.array([-1,1]),'beta':np.array([-1,1])}      # bounds for simulated regression coefficients
    average_expression_prior = np.loadtxt(data_path+'average_expression_mean_and_std.txt')
    
    # bounds for truncated normal distribution from which the average expression levels are drawn
    lb = np.log(0.5)
    ub = np.log(100)
    
    # load estimated normalization constants
    norm_consts_est = np.loadtxt(data_path+'jittered_norm_consts.txt')
    
    # matrices for convergence diagnostics
    PSRF    = np.zeros((n_tests,2*M + 3))
    
    np.random.seed(123*job_id) # random seed job_id based
    for i in range(n_tests):
        # sample input genes from D and for X and log_CX
        log_mux_true = schirm.sim_truncnormal_dist(average_expression_prior[0],average_expression_prior[1],lb,ub,M)
        mux_true     = np.exp(log_mux_true)
        X, log_CX    = schirm.sim_indep_data(mux_true,alpha,c_true,N)
        # sample beta0 and beta and simulate target
        beta0, beta = schirm.sim_beta(M,bounds,Mact)
        y, log_cy   = schirm.sim_output_data(log_CX,c_true,alpha,beta0,beta,N)
        # hierarchical regression
        fit                  = schirm.run_inference(X,y,prior_params,norm_consts_est,stan_model)
        result_p1, result_p2 = schirm.fit2table(fit,'',M)
        PSRF[i,:]            = fit.summary()['summary'][(M+1)*N:,-1]
        # run other methods
        result_reg_p1, result_reg_p2 = schirm.run_standard_reg(X,y,norm_consts_est)
        # combine results
        temp1 = pd.concat([pd.DataFrame(data={'beta_true' : beta}),
                           pd.DataFrame(data={'mux_true' : mux_true}),
                           result_p1,
                           result_reg_p1], axis=1)
        temp2 = pd.concat([pd.DataFrame(index=np.array([0]),data={'beta0_true' : beta0}),
                           result_p2,
                           result_reg_p2], axis=1)
        if i > 0:
            df1 = df1.append(temp1, ignore_index=True)
            df2 = df2.append(temp2, ignore_index=True)
        else:
            df1 = temp1
            df2 = temp2
        print(str(100*(i+1)/n_tests)+'% finished.')
    df1.to_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p1' + '.csv')
    df2.to_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p2' + '.csv')
    np.savetxt(result_path+run_name+'rhat_'   + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '.txt', PSRF)

def run_testing_with_real_data(M,Mact_temp,n_tests,job_id,run_name,
                                    data_path,data_file_name,result_path):

    if Mact_temp == 0:
        Mact = None
    else:
        Mact = Mact_temp

    # load data and normalization constants
    df = pd.read_csv(data_path + data_file_name + '.csv',index_col = 0)
    gene_names = list(df.columns.values)
    norm_consts_est = np.loadtxt(data_path+'GSM1599500_K562_cells_norm_consts.txt')
    N = len(norm_consts_est)

    # list of cell cycle genes
    cell_cyc_genes =['AURKA','E2F5','CCNE1','CDC25A','CDC6','CDKN1A','CDKN3','E2F1','MCM2','MCM6','NPAT','PCNA','BRCA1','BRCA2','CCNG2','CDKN2C','DHFR','PLK1','MSH2','NASP','RRM1','RRM2','TYMS','CCNA2','CCNF','CENPF','TOP2A','BIRC5','BUB1','BUB1B','CCNB1','CCNB2','CDC20','CDC25B','CDC25C','CDKN2D','CENPA','CKS2']

    # remove cell cycle genes from the full gene list
    for name in cell_cyc_genes:
        gene_names.remove(name)

    # load model
    stan_model = schirm.define_model(compile_model = False)
    prior_params = {'beta0_prior_std'   :10.0,
                    'beta_prior_std'    :1.0,
                    'log_mux_prior_mean':-1.64,
                    'log_mux_prior_std' :1.84,
                    'alpha_prior_std'   :0.15}

    # matrices for convergence diagnostics
    PSRF = np.zeros((n_tests,2*(M) + 3))

    np.random.seed(100*job_id)  # job_id based -> different subsets for individual jobs
    for i in range(n_tests):
        #
        if Mact == None:
            if M%2 == 0:
                Mact = int(M/2)
        else:
            Mact = int(np.ceil(M/2)) - np.random.randint(2) 
        target_index  = np.random.choice(len(cell_cyc_genes), 1, replace=False)
        true_indices  = np.random.choice(len(cell_cyc_genes), Mact, replace=False)
        false_indices = np.random.choice(len(gene_names), M - Mact, replace=False)
        target_name        = cell_cyc_genes[target_index[0]]
        true_driver_names  = list( cell_cyc_genes[xxx] for xxx in true_indices )
        false_driver_names = list( gene_names[xxx] for xxx in false_indices )
        X                  = df[true_driver_names+false_driver_names].as_matrix()
        y                  = df[target_name].as_matrix()
        beta = np.zeros(M)
        beta[0:Mact] = 1
        # hierarchical regression
        fit                  = schirm.run_inference(X,y,prior_params,norm_consts_est,stan_model)
        result_p1, result_p2 = schirm.fit2table(fit,'',M)
        PSRF[i,:]            = fit.summary()['summary'][(M+1)*N:,-1]
        # run other methods
        result_reg_p1, result_reg_p2 = schirm.run_standard_reg(X,y,norm_consts_est)
        # combine results
        temp1 = pd.concat([pd.DataFrame(data={'beta_true' : beta}),
                           result_p1,
                           result_reg_p1], axis=1)
        temp2 = pd.concat([result_p2,
                           result_reg_p2], axis=1)
        if i > 0:
            df1 = df1.append(temp1, ignore_index=True)
            df2 = df2.append(temp2, ignore_index=True)
        else:
            df1 = temp1
            df2 = temp2
        print(str(100*(i+1)/n_tests)+'% finished.')
      
    df1.to_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p1' + '.csv')
    df2.to_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p2' + '.csv')
    np.savetxt(result_path+run_name+'rhat_'   + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '.txt', PSRF)

def combine_results(M,Mact_temp,n_jobs,run_name,result_path,result_path_comb):
    job_id = 1
    df1 = pd.read_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p1' + '.csv',index_col=0)
    df2 = pd.read_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p2' + '.csv',index_col=0)
    rhat    = np.loadtxt(result_path+run_name+'rhat_'+ str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '.txt')
    
    for job_id in range(2,n_jobs+1):
        temp1 = pd.read_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p1' + '.csv',index_col=0)
        temp2 = pd.read_csv(result_path+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '_p2' + '.csv',index_col=0)
        rhat_temp = np.loadtxt(result_path+run_name+'rhat_'+ str(M) + '_' + str(Mact_temp) + '_' + str(job_id) + '.txt')
        # append
        df1     = df1.append(temp1, ignore_index=True)
        df2     = df2.append(temp2, ignore_index=True)
        rhat    = np.append(rhat,rhat_temp,axis=0)
        
    # save
    df1.to_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p1' + '.csv')
    df2.to_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p2' + '.csv')
    np.savetxt(result_path_comb+run_name+'rhat_'   + str(M) + '_' + str(Mact_temp) + '.txt', rhat)
    
def illust_AUC_summary(MMM,Mact,run_name1,run_name2,result_path_comb):
    run_name         = run_name1
    n = len(MMM)
    AUC = np.zeros((n,7))
    names = ['SCHiRMp','SCHiRMpm','OLS', 'LASSO','ENET','RIDGE','PCC']
    
    plt.subplot(2,1,1)
    for i in range(n):
        M = MMM[i]
        Mact_temp = Mact[i]
        df = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p1' + '.csv',index_col=0)
        th = 0
        y_true = np.abs(df['beta_true'].as_matrix()) > th
        for j, name in enumerate(names):
            score    = np.abs(df[name].as_matrix())
            AUC[i,j] = schirm.roc_analysis(y_true, score, name, illust = False)
    
    for j, name in enumerate(names):
        plt.plot(range(1,len(MMM)+1),AUC[:,j],'-o',label = name)
    tick_names = []
    for i in range(len(MMM)):
        tick_names.append(str(MMM[i]) + '('+str(Mact[i]) + ')')
    plt.xticks(range(1,len(MMM)+1),tick_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Number of inputs (number of active inputs)')
    plt.ylabel('AUC')
    plt.title('Simulated data')
    ########
    
    run_name         = run_name2
    n = len(MMM)
    AUC = np.zeros((n,7))
    
    plt.subplot(2,1,2)
    for i in range(n):
        M = MMM[i]
        Mact_temp = Mact[i]
        df = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p1' + '.csv',index_col=0)
        th = 0
        y_true = np.abs(df['beta_true'].as_matrix()) > th
        for j, name in enumerate(names):
            score    = np.abs(df[name].as_matrix())
            AUC[i,j] = schirm.roc_analysis(y_true, score, name, illust = False)
    
    for j, name in enumerate(names):
        plt.plot(range(1,len(MMM)+1),AUC[:,j],'-o',label = name)
    tick_names = []
    for i in range(len(MMM)):
        tick_names.append(str(MMM[i]) + '('+str(Mact[i]) + ')')
    plt.xticks(range(1,len(MMM)+1),tick_names)
    plt.xlabel('Number of inputs (number of cell cycle genes)')
    plt.ylabel('AUC')
    plt.title('Real data')
    
def illust_errors(Mall,Mact_all,run_name,result_path_comb):
    n = len(Mall)
    
    names = ['SCHiRMpm','OLS', 'LASSO','ENET','RIDGE']
    names_show = ['SCHiRM','OLS', 'LASSO','ENET','RIDGE']
    
    # regression coefficients
    for i in range(n):
        M = Mall[i]
        Mact_temp = Mact_all[i]
        df = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p1' + '.csv',index_col=0)
        true_vals = df['beta_true'].as_matrix()
        for j, name in enumerate(names):
            error      = np.abs(df[name].as_matrix() - true_vals)**2
            plt.subplot(5,2,2*j+1)
            ind = np.argsort(true_vals)
            plt.scatter(true_vals[ind],error[ind],c='teal',alpha = 0.05)
            plt.title(names_show[j])
            if j == 2:
                plt.ylabel('Squared error')
            if j < 4:
                plt.tick_params(
                                axis='x',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off
            else:
                plt.xlabel('True coefficient')
    
    # intercept
    names = ['beta0_mean','beta0_ols', 'beta0_las', 'beta0_ene', 'beta0_rid']      
    for i in range(n):
        M = Mall[i]
        Mact_temp = Mact_all[i]
        df = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p2' + '.csv',index_col=0)
        true_vals = df['beta0_true'].as_matrix()
        for j, name in enumerate(names):
            error      = np.abs(df[name].as_matrix() - true_vals)**2
            plt.subplot(5,2,2*j+2)
            ind = np.argsort(true_vals)
            plt.scatter(true_vals[ind],error[ind],c='teal',alpha = 0.05)
            if j < 4:
                plt.tick_params(
                                axis='x',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off
            else:
                plt.xlabel('True intercept')
                
def param_est_fig(run_name,result_path_comb):
    n_rows = 4
    n_cols = 5
    
    #Mact= np.array([3,4,4,4,4])
    #MMM = np.array([7,8,12,16,20])
    Mact= np.array([1,1,2,2,3])
    MMM = np.array([2,3,4,5,6])
    n_tests = len(MMM)
    
    fig = plt.figure(figsize=(2*n_cols,8))
    for i in range(n_tests):
        M = MMM[i]
        Mact_temp = Mact[i]
        df1 = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p1' + '.csv',index_col=0)
        df2 = pd.read_csv(result_path_comb+run_name+'res_'    + str(M) + '_' + str(Mact_temp) + '_p2' + '.csv',index_col=0)
    
        # true betas vs est
        plt.subplot(n_rows,n_cols,i+1)
        plt.plot([-1,1],[-1,1],'k')
        plt.scatter(df1['beta_true'].as_matrix(),df1['SCHiRMpm'].as_matrix(),s = 0.2)
        plt.title('M = ' + str(MMM[i]))
        if i == 0:
            plt.xlabel('True $\\beta$')
            plt.ylabel('Est. $\\beta$')
        # true beta0 vs est
        plt.subplot(n_rows,n_cols,i + n_cols + 1)
        plt.plot([-1,1],[-1,1],'k')
        plt.scatter(df2['beta0_true'].as_matrix(),df2['beta0_mean'].as_matrix(),s = 0.2)
        if i == 0:
            plt.xlabel('True $\\beta_0$')
            plt.ylabel('Est. $\\beta_0$')
        
        # true mu vs est
        plt.subplot(n_rows,n_cols,i + 2*n_cols + 1)
        plt.plot([0,100],[0,100],'k')
        plt.scatter(df1['mux_true'].as_matrix(),df1['mux_mean'].as_matrix(),s = 0.6)
        if i == 0:
            plt.xlabel('True $\mu$')
            plt.ylabel('Est. $\mu$')
            
        plt.subplot(n_rows,n_cols,i + 3*n_cols + 1)
        plt.hist(df2['alpha_mean'].as_matrix())
        if i == 0:
            plt.xlabel('$\\alpha$')
            plt.ylabel('Freq.')