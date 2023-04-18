import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
import functions
import emcee

#Getting the data
data = pd.read_csv('/home/michael/Documents/Mcgill_Courses/PHYS 321/PHYS321-final-project/cristiano-ronaldo-stats.csv') # loading up the data retrieved from the websites
goalScored = data['pen-scored'].astype(float) # assigning them varaible names
goalDiff = data['score-diff'].astype(float)
saveRate = data['goal-keeper-success-rate'].astype(float)

#Global variables
numBetas = 300 #This is the number of betas we are testing

num_iter = 1000
ndim = 3 # number of parameters
nwalkers = ndim*18
initial_pos = np.array((2.5, -15, 2)) + 0.01 * np.random.randn(nwalkers, ndim)

sampler3 = emcee.EnsembleSampler(nwalkers, ndim, functions.log_postNew, args=(np.array([saveRate,goalDiff]),goalScored))
sampler3.run_mcmc(initial_pos, num_iter, progress=True);

flat_samples = sampler3.get_chain(discard=100, thin=15, flat=True)

#Now lets plot a 3d logistic regression

# importing required libraries
from random import randrange
from mpl_toolkits.mplot3d import Axes3D
inds = np.random.randint(len(flat_samples), size=100)

x0 = np.linspace(0, 1., len(saveRate))
x1 = np.linspace(np.min(goalDiff), np.max(goalDiff), len(goalDiff))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(saveRate, goalDiff, goalScored, color='red')
ax.set_xlabel("Goalkeeper Save Rate")
ax.set_ylabel("Goal Difference")
ax.set_zlabel("P(Scoring)")

for ind in inds:
    #ignore every two so the graph isn't too crowded
    if(randrange(3)==1):
        continue
    sample = flat_samples[ind]
    X, Y = np.meshgrid(x0,x1)
    Z = functions.pNew([sample[0],sample[1],sample[2]], [X,Y])
    ax.plot_surface(X,Y,Z, alpha=0.01, color='red')


plt.show()