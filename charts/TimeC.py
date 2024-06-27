#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:12:13 2023

@author: mohammadaminbasiri
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import factorial

a = np.linspace(0.0, 1.0, 100) # n is connectivity




#Lambda
L = 5

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.plot(a, np.exp(-(L*a)/1.0), color='red', label='k = 1', linewidth=3)
ax.plot(a, np.exp(-(L*a)/2.0), color='blue', label='k = 2', linewidth=3)
ax.plot(a, np.exp(-(L*a)/3.0), color='green', label='k = 3', linewidth=3)
ax.plot(a, np.exp(-(L*a)/4.0), color='orange', label='k = 4', linewidth=3)
ax.plot(a, np.exp(-(L*a)/5.0), color='violet', label='k = 5', linewidth=3)
ax.plot(a, np.exp(-(L*a)/20.0), color='cyan', label='k = 20', linewidth=3)
# ax.plot(n, 3**n, color='orange', label='3^n', linewidth=3)
# ax.plot(n, factorial((n)), color='violet', label='n!', linewidth=3)
# ax.plot(n, n*np.log2(n), color='cyan', label='nlogn', linewidth=3)
ax.legend(loc='upper right', shadow=True)
# plt.savefig('accuracy.png')
ax.set_title('Comparision')
ax.set_xlabel('Connectivity')
ax.set_ylabel('F(a)')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0, right=1)
ax.grid(True)
#plt.yscale("log")
plt.show()



# %%
### dsd

#Lambda
L = 8


fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.plot(a, 1-np.exp(-1.0/(L*a)), color='red', label='k = 1', linewidth=3)
ax.plot(a, 1-np.exp(-2.0/(L*a)), color='blue', label='k = 2', linewidth=3)
ax.plot(a, 1-np.exp(-3.0/(L*a)), color='green', label='k = 3', linewidth=3)
ax.plot(a, 1-np.exp(-4.0/(L*a)), color='orange', label='k = 4', linewidth=3)
ax.plot(a, 1-np.exp(-5.0/(L*a)), color='violet', label='k = 5', linewidth=3)
ax.plot(a, 1-np.exp(-20.0/(L*a)), color='cyan', label='k = 20', linewidth=3)
# ax.plot(n, 3**n, color='orange', label='3^n', linewidth=3)
# ax.plot(n, factorial((n)), color='violet', label='n!', linewidth=3)
# ax.plot(n, n*np.log2(n), color='cyan', label='nlogn', linewidth=3)
ax.legend(loc='upper right', shadow=True)
# plt.savefig('accuracy.png')
ax.set_title('Comparision')
ax.set_xlabel('Connectivity')
ax.set_ylabel('F(a)')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0, right=1)
ax.grid(True)
#plt.yscale("log")
plt.show()

# %%
### Sigmoid

#Lambda
L = 10


fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.plot(a, 1/(1+np.exp(-1.0/(L*a))), color='red', label='k = 1', linewidth=3)
ax.plot(a, 1/(1+np.exp(-2.0/(L*a))), color='blue', label='k = 2', linewidth=3)
ax.plot(a, 1/(1+np.exp(-3.0/(L*a))), color='green', label='k = 3', linewidth=3)
ax.plot(a, 1/(1+np.exp(-4.0/(L*a))), color='orange', label='k = 4', linewidth=3)
ax.plot(a, 1/(1+np.exp(-5.0/(L*a))), color='violet', label='k = 5', linewidth=3)
ax.plot(a, 1/(1+np.exp(-20.0/(L*a))), color='cyan', label='k = 20', linewidth=3)
# ax.plot(n, 3**n, color='orange', label='3^n', linewidth=3)
# ax.plot(n, factorial((n)), color='violet', label='n!', linewidth=3)
# ax.plot(n, n*np.log2(n), color='cyan', label='nlogn', linewidth=3)
ax.legend(loc='upper right', shadow=True)
# plt.savefig('accuracy.png')
ax.set_title('Comparision')
ax.set_xlabel('Connectivity')
ax.set_ylabel('F(a)')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0, right=1)
ax.grid(True)
#plt.yscale("log")
plt.show()

# %%
### sigmoid

#Lambda
L = 10


fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.plot(a, 1/(1+np.exp((L*(a-0.5)))), color='red', label='k = 1', linewidth=3)
ax.plot(a, 1/(1+np.exp((L*a-1))), color='blue', label='k = 2', linewidth=3)
ax.plot(a, 1/(1+np.exp((L*a))), color='green', label='k = 3', linewidth=3)
ax.plot(a, 1/(1+np.exp((L*a))), color='orange', label='k = 4', linewidth=3)
ax.plot(a, 1/(1+np.exp((L*a))), color='violet', label='k = 5', linewidth=3)
ax.plot(a, 1/(1+np.exp((L*a))), color='cyan', label='k = 20', linewidth=3)
# ax.plot(n, 3**n, color='orange', label='3^n', linewidth=3)
# ax.plot(n, factorial((n)), color='violet', label='n!', linewidth=3)
# ax.plot(n, n*np.log2(n), color='cyan', label='nlogn', linewidth=3)
ax.legend(loc='upper right', shadow=True)
# plt.savefig('accuracy.png')
ax.set_title('Comparision')
ax.set_xlabel('Connectivity')
ax.set_ylabel('F(a)')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0, right=1)
ax.grid(True)
#plt.yscale("log")
plt.show()

# %%


#python3 main.py -m umap 1000,64,30 ../data/raw_tseries/synthetic_tseries/

import numpy as np

# Load the .npy file
data = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/results/raw_tseries_20_rectangle_1_k5_pca_32_2024_6_17_23_31_35/clusters_pca.npy', allow_pickle=True)
data1 = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/results/raw_tseries_20_rectangle_1_k5_pca_32_2024_6_17_23_31_35/states_pca.npy', allow_pickle=True)

# Now you can use 'data' as a NumPy array
print(data)