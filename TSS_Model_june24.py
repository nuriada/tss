#!/usr/bin/env python
# coding: utf-8

# Import packages
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import warnings
# Omit all warnings
warnings.filterwarnings("ignore")


# # Key parameters of simulation
## G: The heat pump is assumed to have a "Gütegrad" (G) of 0.4.
## TDHW: Target temperature of DHW / Maximum temperature of STES and heat pump outlet temperature is set to 60° C.
## TSH: Target temperature of SH 
## COP: We use a dynamic COP based on the outdoor temperature (Ta) and the target temperatures of DHW and SH.
## Storage Sizes: The storage sizes in Wh range from 0 to 500000 Wh
## Ta: Outside Temperature based on the measurement date.
## PV: Potential energy production through photovoltaic 
## Elec: electrical consumption of the bulding, excluding the electrical consumption of for the heating in watt
## DHW: Hot water demand of the buildilding in Watt
## SH: Speace heating demand of the building in Watt
## TaC: Outside temperature on degrees C
## potHeat: Potential heat that can be generated after meeting current electricity demand

G = 0.4
TDHW = 273.15+60
TSH = 273.15+40
n_years=2
storage_sizes_1=range(0,6001,1000)
storage_sizes_2=[7500,10000,15000,20000,30000,50000,100000,250000]
storage_sizes_3=range(0,4500001, 500000)
storage_sizes=list(storage_sizes_1)+list(storage_sizes_2)+list(storage_sizes_3)
storage_sizes=list(dict.fromkeys(storage_sizes))[1:]
direc='.'

# # Reading Input data and data organisation
# Read in csv file with outside temperature data
data_temp = pd.read_excel("input-data/data_temperature.xlsx")
data_temp['Ta'] = data_temp['TaC']+273.15

# Read in csv file with building load data and mearge it with data_temp
data=pd.read_csv("input-data/Daten-sheet.csv",delimiter=";")
data=data.dropna()
# Rename columns
data.columns=["Date","PV","Elec","DHW","SH"]
data["Day"]=data["Date"].str[:5]
data=data[['Date', 'Day', 'PV', 'Elec', 'DHW', 'SH']]
data=pd.merge(data, data_temp[['Date','Ta','TaC']])
# Add Heat= DHW+SH 
data['Heat']=data.loc[:,'DHW']+data.loc[:,'SH']

#Factor for Sensitivity
Factor_PV=1
# Calculate excess PV as difference of PV production and electricity used
excessPV = (data.loc[:, "PV"] * Factor_PV) - data.loc[:, "Elec"]
excessPV[excessPV<0]=0
data['excessPV']=excessPV
data=data[['Date', 'Day', 'PV', 'Elec', 'DHW', 'SH', 'Heat', 'TaC', 'Ta','excessPV']]

# Computation of COP
def COP_computation(G, Ttarget, Ta):
    COP=G*Ttarget/(Ttarget-Ta)
    return(COP)

# Compute COP for Ttargets
data['COPdhw']=COP_computation(G,TDHW,data['Ta'])
data['COPsh']=COP_computation(G,TSH,data['Ta'])

# Transform the excess PV to potential DHW production using its COP
data['potDHW'] = data['COPdhw'] * ( data['excessPV'])
# Excess DHW as difference of potential DHW production and DHW demand
data['excessDHW']=data['potDHW']-data['DHW']
# Consider the improved efficiency of SH by calculating excessSH = excessDHW*COPsh/COPdhw
data['excessSH']=data['excessDHW']*data['COPsh']/data['COPdhw']
# In case the excess SH is negative (implying that excess DHW is negative), put it to 0
data.loc[data['excessSH']<0,'excessSH']=0
# Consider the excess heat as the difference of excess SH and the SH demand
data['excessHeat']=data['excessSH']-data['SH']
# In case excess SH is 0, consider excess Heat as the difference of excess DHW (which is negative) and SH demand
data.loc[data['excessSH']==0,'excessHeat']=data['excessDHW']-data['SH']

# Put already the storage level and uncovered heat for capacity=0 (the uncovered demand corresponds to the unmet demand)
data['storageLevel_0'] = 0
data['uncoveredHeatAfterStorage_0'] = data['unmetHeat'].apply(lambda x:0 if x<0 else x)
# replicate the input information for one more year so that the level of storage for (big) capacities is not 0 in April
data_concat=pd.concat([data]*n_years)
data_concat=data_concat.reset_index(drop=True)
exheat=data_concat['excessHeat'].copy().values
exheat=pd.Series(exheat)

# In exheat, replace the '0' for the value immediately above it so that the '0' are not identified as a sign change for the
# charge/discharge simulation
exheatnozero=exheat.copy()
exheatzero=np.where(exheat==0)[0]
while np.any(exheatnozero[exheatzero]==0):
    exheatnozero[exheatzero]=exheatnozero[exheatzero-1]
    exheatzero=np.where(exheatnozero==0)[0]
# Identify positions where there is a change from negative to positive (i.e., charge)
neg_to_pos = np.where(np.diff(np.sign(exheatnozero)) > 0)[0] + 1
# Identify positions where there is a change from positive to negative (i.e., discharge)
pos_to_neg = np.where(np.diff(np.sign(exheatnozero)) < 0)[0] + 1
# positions where to begin charging (0 and even positions) or discharging (uneven positions)
begin=np.concatenate((neg_to_pos, pos_to_neg))
begin=np.sort(begin)
# positions where to end charging (0 and even positions) or discharging (uneven positions)
end=np.roll(begin, -1)
end[-1]=len(exheat)


# # Loss function
# Losses based on https://www.npro.energy/main/en/help/heat-storage-loss :Specific loss rates vary depending on storage tank size, with estimates such as 10% per day for small storage tanks of 0.75 m³ and up to 35% per year for very large storage tanks of 70,000 m³
# Energy_density = 50  kWh/m3
palette = sns.color_palette("Blues_d")
data_points = np.array([
    [37.5, 0.00438039942691804],
    [1500, 0.0018578017402936],
    [15000, 0.000438905801230516],
    [150000, 0.000225695262405012],
    [3500000, 0.0000491749228478389]
])

x_data, y_data = data_points[:, 0], data_points[:, 1]

def power_function(x, a, b):
    return a * x**b

params, _ = curve_fit(power_function, x_data, y_data)
energies_plot = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 500)

plt.figure(figsize=(4, 4))
plt.subplots_adjust(left=0.17)

sns.set_palette("Blues_d")
plt.scatter(x_data, y_data,  marker='x', label="Data npro.energy")
plt.plot(energies_plot, power_function(energies_plot, *params), label="Losses as function",color=palette[1])
plt.xscale('log') 
plt.yscale('log') 
plt.xlabel("TES capacity (kWh)")
plt.ylabel("Hourly loss rate (%)")
plt.legend()
plt.text(1e4, 0.001, r'$y\approx$' + f'{params[0]:.4f}' + r'$\cdot x$^' + f'{params[1]:.4f}', fontsize=9)
plt.show()


def calculate_loss(storage_level, loss_mode="npro", fraction=0.001):
    if loss_mode == "npro":
        loss = fraction* storage_level
    else:
        loss = 0.0

    return loss


# ### Function for loading process
def charge_with_loss(exheat, storage_level, storage_size, begin, end, pos):
    # excessHeat: Output variable of potentially still available excess heat during simulation window between begin[pos] and end[pos]
    bp = begin[pos]
    ep = end[pos]
    lev = storage_level[bp - 1]
    print(f"Charge with loss: Initial 'begin': {bp}, 'end': {ep}, storage_level={lev}")
    losses = []
    excessHeat=[]
    
    for i in range(bp, ep):
        current_level=storage_level[i-1]+exheat[i]
        loss = calculate_loss(current_level,fraction=params[0]*np.power(storage_size,params[1]) )
        losses.append(loss)
        storage_level[i] = max(current_level - loss, 0)

        if storage_level[i]>storage_size:
            #storage full -> fill excess heat   
            excessHeat.append(storage_level[i]-storage_size)
            storage_level[i]=storage_size
        else:
            # storage was not full
            excessHeat.append(0)
 
    return storage_level, losses, excessHeat
 


# ### Function for discharging
def discharge_with_loss(exheat, storage_level,  storage_size, begin, end, pos, uncov):
    bp = begin[pos]
    ep = end[pos]
    losses = []
    # here no excess heat is calculated because it is expected to be negative 
    # excessHeat=[]
    
    for i in range(bp, ep):
        print("Current storage level",i, " at ",storage_level[i-1], " demand ", exheat[i]," heat ",data.loc[i%8760,"Heat"], " pot ",data.loc[i%8760,"potDHW"])
        
        previous_storage_level=storage_level[i-1]
        loss = calculate_loss(previous_storage_level,fraction=params[0]*np.power(storage_size,params[1]))
        losses.append(loss)
        previous_storage_level=max(previous_storage_level-loss,0)

        # split the consumption 
        #  - exheat[i] is the consumption at the moment
        if (-exheat[i]<previous_storage_level):
            # feed everythign with storage
            storage_level[i]=previous_storage_level+exheat[i] # exheat[i] is the NEGATIVE demand
            uncov[i]=0.0 # No heat is uncovered
        else :
            # storage empty
            storage_level[i]=0.0
            uncov[i]=-exheat[i]-previous_storage_level

    return storage_level, uncov, losses


# ## Main simulation loop

loss_data = pd.DataFrame()
storage_level0=pd.Series([0] * len(exheat))
surplusHeat_0=pd.Series([0] * len(exheat))
uncov0=storage_level0.copy()
uncov0[0:8]=data.loc[0:7,'uncoveredHeatAfterStorage_0']
storage_sizes=list(storage_sizes_1)+list(storage_sizes_2)+list(storage_sizes_3)
storage_sizes=list(dict.fromkeys(storage_sizes))[1:]

for storage_size in storage_sizes:
    storage_level = storage_level0.copy()
    uncov = uncov0.copy()
    surplusHeat=surplusHeat_0.copy()    
    total_losses=np.zeros(len(storage_level))
    pos = 0

    print(pos, len(begin))
    while pos < (len(begin) - 1):
        storage_level, new_losses,newSurplusHeat = charge_with_loss(exheat, storage_level, storage_size, begin, end, pos)
        total_losses[begin[pos]:end[pos]]=new_losses
        surplusHeat[begin[pos]:end[pos]]=newSurplusHeat
        [storage_level, uncov,new_losses] = discharge_with_loss(exheat, storage_level, storage_size, begin, end, pos+1, uncov)    
        total_losses[begin[pos+1]:end[pos+1]]=new_losses       
        pos += 2  

    loss_data[f'losses_size_{storage_size}'] = total_losses
    loss_data[f'storageLevel_size_{storage_size}'] = storage_level
    loss_data[f'uncov_demand_losses_{storage_size}'] = uncov
    loss_data[f'surplus_heat_{storage_size}'] = surplusHeat   


# # Compute HSS
# First and second year data
data_firstyear_loss = loss_data[:8760] 
data_secondyear_loss = loss_data[8760:] 

# Calculation HSS under losses
total_heat_demand = data['Heat'].sum()
hss_losses = {}
for size in storage_sizes:
    uncovered_heat = data_secondyear_loss[f'uncov_demand_losses_{size}'].sum()
    hss_losses[size] = 100 * (1 - uncovered_heat / total_heat_demand)

