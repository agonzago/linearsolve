import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib # Modern alternative for path manipulationimport pathlib # Modern alternative for path manipulation
import sys
import os

# --- Add the project root to sys.path ---
# Get the path to the directory containing this script (examples/gpm_try)
script_dir = pathlib.Path(__file__).parent.resolve()
# Get the path to the project root (which is two levels up from examples/gpm_try)
project_root = script_dir.parent.parent.resolve()

# Add the project root to the beginning of the Python path
if str(project_root) not in sys.path:
    print(f"Adding project root to Python path: {project_root}")
    sys.path.insert(0, str(project_root))
# --- End of path modification ---

# Import numpy, pandas, linearsolve, matplotlib.pyplot

import linearsolve as ls

plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'

#Parameters from readmodel.m
b1 = 0.7;          # Output persistence
b4 = 0.7;          # MCI weight
a1 = 0.5;          # Inflation persistence
a2 = 0.1;          # RMC passthrough
g1 = 0.7;         # Interest rate smoothing
g2 = 0.3;          # Inflation response
g3 = 0.25;         # Output gap response
rho_L_GDP_GAP  =0.75;
rho_DLA_CPI   =0.75;
rho_rs      =0.75;
rho_rs2      =0.001;

parameters = pd.Series({
    'b1': b1,
    'b4': b4,
    'a1': a1,
    'a2': a2,
    'g1': g1,
    'g2': g2,
    'g3': g3,
    'rho_L_GDP_GAP': rho_L_GDP_GAP,
    'rho_DLA_CPI': rho_DLA_CPI,
    'rho_rs': rho_rs   
}) 
    

def equations(variables_forward,variables_current,parameters):
    p = parameters
    
    # Variables
    fwd = variables_forward
    cur = variables_current

    #shocks
    SHK_L_GDP_GAP = rho_L_GDP_GAP*cur.RES_L_GDP_GAP-fwd.RES_L_GDP_GAP 
    SHK_DLA_CPI  =  rho_DLA_CPI*cur.RES_DLA_CPI - fwd.RES_DLA_CPI  
    #SHK_RS = rho_rs*cur.RES_RS + rho_rs2*cur.RES_RS_lag - fwd.RES_RS  
    SHK_RS = rho_rs*cur.RES_RS - fwd.RES_RS

    # Aggregate demand
    eq4 = (1-p.b1)*fwd.L_GDP_GAP + p.b1*cur.L_GDP_GAP_lag - p.b4*fwd.RR_GAP + cur.RES_L_GDP_GAP - cur.L_GDP_GAP 


    ## Core Inflation
    eq5 = p.a1*cur.DLA_CPI_lag + (1-p.a1)*fwd.DLA_CPI + p.a2*cur.L_GDP_GAP + cur.RES_DLA_CPI - cur.DLA_CPI 


    ## Monetary policy reaction function
    eq6 = p.g1*cur.RS_lag + (1-p.g1)*(fwd.DLA_CPI + p.g2*fwd.DLA_CPI_led2  + p.g3*cur.L_GDP_GAP) + cur.RES_RS - cur.RS 

    ## Interesrt rate gap
    eq7 =  cur.RS - fwd.DLA_CPI - cur.RR_GAP


    #Auxiliary equations
    eq8 = fwd.L_GDP_GAP_lag - cur.L_GDP_GAP
    eq9 = fwd.DLA_CPI_lag - cur.DLA_CPI
    eq10 = fwd.RS_lag - cur.RS
    #eq11 = fwd.RES_RS_lag - cur.RES_RS
    eq11 = cur.DLA_CPI_led - fwd.DLA_CPI
    eq12 = cur.DLA_CPI_led2 - fwd.DLA_CPI_led
    # eq13 = cur.DLA_CPI_led2 - fwd.DLA_CPI_led
    # eq14 = cur.DLA_CPI_led3 - fwd.DLA_CPI_led2

    return np.array([
            SHK_L_GDP_GAP,
            SHK_DLA_CPI,
            SHK_RS,            
            eq4,
            eq5,
            eq6,
            eq7,
            eq8,
            eq9,
            eq10,
            eq11,
            eq12                           
        ])

# variables = ['RES_L_GDP_GAP','RES_DLA_CPI','RES_RS',
#             'L_GDP_GAP','DLA_CPI','RS','RR_GAP',
#             'L_GDP_GAP_lag','DLA_CPI_lag','RS_lag','RES_RS_lag',
#             'DLA_CPI_led','DLA_CPI_led2','DLA_CPI_led3'
#         ]

variables = ['RES_L_GDP_GAP','RES_DLA_CPI','RES_RS',
            'L_GDP_GAP','DLA_CPI','RS','RR_GAP',
            'L_GDP_GAP_lag','DLA_CPI_lag','RS_lag',
            'DLA_CPI_led', 'DLA_CPI_led2'            
        ]


gpm = ls.model(equations=equations,
            n_states=6,
            n_exo_states = 3,
            variables=variables,
            parameters=parameters)

gpm.set_ss(np.zeros(12))


gpm.linear_approximation()


gpm.solve_klein(gpm.a,gpm.b)

gpm.impulse(T=40,t0=1,shocks=[1,1,1],normalize=False)


# Create the figure and axes
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(3,1,1)
# ax2 = fig.add_subplot(3,1,2)
# ax3 = fig.add_subplot(3,1,3)

# Plot commands
(gpm.irs['e_RES_RS'][['RS','L_GDP_GAP','DLA_CPI','RR_GAP']]*100).plot(lw='5',alpha=0.5,grid=True,title='Monetary Policy',ax=ax1,legend=False)
#(gpm.irs['e_RES_DLA_CPI'][['RES_DLA_CPI','L_GDP_GAP','DLA_CPI','RR_GAP']]*100).plot(lw='5',alpha=0.5,grid=True,title='Inflation shock',ax=ax2,legend=False)
#(gpm.irs['e_RES_L_GDP_GAP'][['RES_L_GDP_GAP','L_GDP_GAP','DLA_CPI','RR_GAP']]*100).plot(lw='5',alpha=0.5,grid=True,title='Demand',ax=ax3).legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=5);