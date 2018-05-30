#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import schirm_tools as schirm

path = './data/'
data_file_name = 'GSM1599500_K562_cells'

# load full data
df = pd.read_csv(path+data_file_name+'.csv',index_col = 0, skiprows=1, header=None)
df = df.T

#
lb = np.log(0.5)
ub = np.log(100)
df, c, m_mean, m_std = schirm.compute_norm_constants_and_mean_expression_prior(df,lb,ub,illust = True)

# save results
df.to_csv(path+data_file_name+'_zeros_removed.csv')
np.savetxt(path+'average_expression_mean_and_std.txt', np.array([m_mean,m_std]))
np.savetxt(path+data_file_name+'_norm_consts.txt', c)

alpha = 0.3
N     = len(c)
norm_consts_est = schirm.compute_jittered_norm_constants(m_mean,m_std,c,N,alpha,Mtot = 5000)
np.savetxt(path+'jittered_norm_consts.txt', norm_consts_est) 

plt.subplot(2,2,4)
plt.plot(np.log(c),np.log(norm_consts_est),'.')
plt.xlabel('True norm. const.')
plt.ylabel('Est norm. const.')

plt.show()