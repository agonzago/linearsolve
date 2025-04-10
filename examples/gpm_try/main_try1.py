
# Import numpy, pandas, linearsolve, matplotlib.pyplot
import numpy as np
import pandas as pd
#import linearsolve as ls
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'


# Input model parameters
beta = 0.99
sigma= 1
eta  = 1
omega= 0.8
kappa= (sigma+eta)*(1-omega)*(1-beta*omega)/omega

rhor = 0.9
phipi= 1.5
phiy = 0

rhog = 0.5
rhou = 0.5
rhov = 0.9

Sigma = 0.001*np.eye(3)

# Store parameters
parameters = pd.Series({
    'beta':beta,
    'sigma':sigma,
    'eta':eta,
    'omega':omega,
    'kappa':kappa,
    'rhor':rhor,
    'phipi':phipi,
    'phiy':phiy,
    'rhog':rhog,
    'rhou':rhou,
    'rhov':rhov
})


# Define function that computes equilibrium conditions
def equations(variables_forward,variables_current,parameters):
    
    # Parameters 
    p = parameters
    
    # Variables
    fwd = variables_forward
    cur = variables_current
    
    # Exogenous demand
    g_proc =  p.rhog*cur.g - fwd.g
    
    # Exogenous inflation
    u_proc =  p.rhou*cur.u - fwd.u
    
    # Exogenous monetary policy
    v_proc =  p.rhov*cur.v - fwd.v
    
    # Euler equation
    euler_eqn = fwd.y -1/p.sigma*(cur.i-fwd.pi) + fwd.g - cur.y
    
    # NK Phillips curve evolution
    phillips_curve = p.beta*fwd.pi + p.kappa*cur.y + fwd.u - cur.pi
    
    # interest rate rule
    interest_rule = p.phiy*cur.y+p.phipi*cur.pi + fwd.v - cur.i
    
    # Fisher equation
    fisher_eqn = cur.i - fwd.pi - cur.r
    
    
    # Stack equilibrium conditions into a numpy array
    return np.array([
            g_proc,
            u_proc,
            v_proc,
            euler_eqn,
            phillips_curve,
            interest_rule,
            fisher_eqn
        ])

# Initialize the nk model

nk = model(equations=equations,
            n_states=3,
            n_exo_states = 3,
            variables=['g','u','v','i','r','y','pi'],
            parameters=parameters)

# Set the steady state of the nk model
nk.set_ss([0,0,0,0,0,0,0])

# Find the linear approximation around the non-stochastic steady state
nk.linear_approximation()

# Solve the nk model
nk.solve_klein(nk.a,nk.b)

# Compute impulse responses
nk.impulse(T=11,t0=1,shocks=[0.01,0.01,0.01],normalize=False)

# Create the figure and axes
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

# Plot commands
(nk.irs['e_g'][['g','y','i','pi','r']]*100).plot(lw='5',alpha=0.5,grid=True,title='Demand shock',ax=ax1,legend=False)
(nk.irs['e_u'][['u','y','i','pi','r']]*100).plot(lw='5',alpha=0.5,grid=True,title='Inflation shock',ax=ax2,legend=False)
(nk.irs['e_v'][['v','y','i','pi','r']]*100).plot(lw='5',alpha=0.5,grid=True,title='Interest rate shock',ax=ax3).legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=5);