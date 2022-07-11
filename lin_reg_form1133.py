# Adapted from a webpage on linear regression
# http://connor-johnson.com/2014/02/18/linear-regression-with-python/

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import statsmodels.formula.api as sm
import scipy, scipy.stats
import matplotlib.pyplot as plt

df = pd.read_table('table_rpnsZ_1133.txt', sep = '\s+')

plt.scatter(df.rpns, df.sigma, marker='.', edgecolor='k',
    facecolor='none', alpha=0.5)
plt.xlabel('rpns')
plt.ylabel('sigma')
plt.show()
plt.savefig('rpns_v_sigma.png', fmt='png', dpi=125)

                    # set up linear regression
Y = df.rpns 
X = df[['sigma', 'sigma_n/2', 'sigma_n/3', 'sigma_n/4', 'sigma_n/6', 'sigma_n/12']] 
result = sm.OLS(Y,X).fit()
print(result.summary())

                   # get the F-statistic
N = result.nobs
P = result.df_model
dfn, dfd = P, N - P - 1                                       
F = result.mse_model / result.mse_resid
p = 1.0 - scipy.stats.f.cdf(F, dfn, dfd)
print ('F-statistic: {:.3f},  p-value: {:.5f}'.format(F,p))

                   # find the log-likelihood
N = result.nobs
SSR = result.ssr
s2 = SSR / N
L = (1.0 / np.sqrt(2*np.pi*s2)) ** N * np.exp(-SSR/(s2*2.0) )
print('ln(L) = ', np.log(L))

