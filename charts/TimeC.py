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
clusters = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/results/raw_tseries_10_rectangle_2_k5_pca_32_2024_6_27_17_33_57/clusters_pca.npy', allow_pickle=True) #raw_tseries_10_rectangle_2_k5_pca_32_2024_6_27_17_33_57
states = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/results/raw_tseries_10_rectangle_2_k5_pca_32_2024_6_27_17_33_57/states_pca.npy', allow_pickle=True) #raw_tseries_20_rectangle_1_k3_pca_32_2024_6_24_21_51_39
dwell_time = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/results/raw_tseries_10_rectangle_2_k5_pca_32_2024_6_27_17_33_57/dwell_time_pca.npy', allow_pickle=True)
fractional_occupancy = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/results/raw_tseries_10_rectangle_2_k5_pca_32_2024_6_27_17_33_57/fractional_occupancy_pca.npy', allow_pickle=True)
dfc = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/data/raw_tseries_20_rectangle_1/dfc.npy', allow_pickle=True)
#clusters = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/clusters.npy', allow_pickle=True)
dfc1 = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/data/raw_tseries_10_rectangle_2/dfc.npy', allow_pickle=True)

#data1 = np.load('/Users/mohammadaminbasiri/Documents/GitHub/dFC_DimReduction/results/raw_tseries_20_rectangle_1_k5_pca_32_2024_6_17_23_31_35/states_pca.npy', allow_pickle=True)

# %%




import matplotlib.pyplot as plt

# Sample array of numbers
# numbers = [1.593520555514270, 1.4412422109424300, 1.3016623351424400, 1.1907764174991700,
#            1.0752995355082800, 0.9941440509984930, 0.9778597641843530, 0.9741731403472720, 
#            0.9127729382583480, 0.8727873125634760, 0.8993432706601190
#            ]
numbers = [1.5397414274026300, 1.3034047511900900, 1.2160962933968600, 
           1.0767133774023200, 1.060528194055300, 0.9029452686424080, 
           0.8866463584105970, 0.8306651051390010, 0.7522525917551150, 
           0.7761520696722300,	0.7541368101724650
           ]

numbers = [1.5183487849688900,	0.8440107406271280,	0.8788024400626790,	0.901829881727107,
           0.9041634732997160,	0.9170466971878560,	0.8790588731199640,	0.8885848113865140,	
           0.8802309027394770,	0.8183189689604550,	0.8759076983754230
           ]

# Get the indices (x-axis values)
indices = range(2, len(numbers)+2)  # Creates a list of indices from 0 to length-1

# Plot the line
plt.plot(indices, numbers)

# Set labels for the axes
plt.xlabel("Index")
plt.ylabel("Value")

# Add a title to the plot (optional)
plt.title("Line Plot of Numbers")

# Display the plot
plt.show()