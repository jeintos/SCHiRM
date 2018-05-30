#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import schirm_tools as schirm

run_name       = 'real'
data_path      = './DATA/'
data_file_name = 'GSM1599500_K562_cells_zeros_removed' # we use data to simulate realistic data
result_path    = './RESULTS/'

Mfalse = 3
Mtrue  = 3
M = Mfalse + Mtrue

# load model and specify priors
stan_model   = schirm.define_model(compile_model = False)
prior_params = {'beta0_prior_std'   :10.0,
                'beta_prior_std'    :1.0,
                'log_mux_prior_mean':-1.64,
                'log_mux_prior_std' :1.84,
                'alpha_prior_std'   :0.15}

# load data and normalization constants
df = pd.read_csv(data_path + data_file_name + '.csv',index_col = 0)
gene_names = list(df.columns.values)
norm_consts_est = np.loadtxt(data_path+'GSM1599500_K562_cells_norm_consts.txt')
N = len(norm_consts_est)

# list of cell cycle genes
cell_cyc_genes = ['AURKA','E2F5','CCNE1','CDC25A','CDC6','CDKN1A','CDKN3','E2F1','MCM2','MCM6','NPAT','PCNA','BRCA1','BRCA2','CCNG2','CDKN2C','DHFR','PLK1','MSH2','NASP','RRM1','RRM2','TYMS','CCNA2','CCNF','CENPF','TOP2A','BIRC5','BUB1','BUB1B','CCNB1','CCNB2','CDC20','CDC25B','CDC25C','CDKN2D','CENPA','CKS2']
# remove cell cycle genes from the full gene list
for name in cell_cyc_genes:
    gene_names.remove(name)

np.random.seed(142) # job_id based random seed

#target_index  = np.random.choice(len(cell_cyc_genes), 1, replace=False)
#true_indices  = np.random.choice(len(cell_cyc_genes), Mtrue, replace=False)
#false_indices = np.random.choice(len(gene_names), Mfalse, replace=False)
target_index  = np.array([30])
true_indices  = np.array([16,6,3])
false_indices = np.array([22520, 10772,  15852])
target_name        = cell_cyc_genes[target_index[0]]
true_driver_names  = list( cell_cyc_genes[xxx] for xxx in true_indices )
false_driver_names = list( gene_names[xxx] for xxx in false_indices )
X                  = df[true_driver_names+false_driver_names].as_matrix()
y                  = df[target_name].as_matrix()
beta = np.zeros(M)
beta[0:Mtrue] = 1

print(X.mean(axis = 0))
# hierarchical regression
fit                  = schirm.run_inference(X,y,prior_params,norm_consts_est,stan_model)
result_p1, result_p2 = schirm.fit2table(fit,'',M)
psrf                 = fit.summary()['summary'][(M+1)*N:,-1]

schirm.illust_post_summary_A(fit, M,true_params=None)