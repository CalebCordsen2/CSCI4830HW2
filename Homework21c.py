
# ## Homework 2 Problem 1c
# ### Mathematical & Computational Modeling of Infectious Diseases
# #### Caleb Cordsen


import numpy as np
import matplotlib.pylab as plt
import math
from scipy.optimize import fsolve

# Problem 1c
# Define a function maxAbError that takes in a deltaT and will calculate the max|IEuler(t) - IAnalytical(t)| using the deltaT.
# Because other initial start up parameters were not provided to us I just carried over the same ones from problem 1a i.e.
# beta = 3, gamma = 2, s0 = 0.99, i0=0.01
def maxAbError(deltaT):
    # Set up some initial constants like rNot and constant components of the analytical solution to simplify equation later
    Beta = 3
    Gamma = 2
    rNot = Beta/Gamma
    numeratorA = 1-(1/rNot)
    eConstant = (numeratorA-0.01) / 0.01
    exponent = Beta-Gamma
    # Set the currMax to be a negative number so it gets replaced with the first calculated number (abs value will always produce a positive)
    currMax = -1
    # Set up variables sPrev and iPrev which at the start will be the initial s0 and i0 values
    sPrev = 0.99
    iPrev = 0.01
    # Set up a timeCounter
    timeCounter = 0
    # I chose to simulate until time unit 10,000 sort of arbitraly. I started with 1000, then 10,000 and then 1,000,000 during testing
    # They all produced the same values so I chose 10,000 because it is pretty large but it was still pretty fast for computation while 1,000,000 took a while
    while timeCounter<=10000:
        # Increment the timeCounter by deltaT
        timeCounter += deltaT
        # Calculate new i,s from Euler as well as a new i for Analytical
        iNewE = iPrev+deltaT*(Beta*sPrev*iPrev-Gamma*iPrev)
        sNewE = sPrev+deltaT*(-Beta*sPrev*iPrev+Gamma*iPrev)
        # Time counter will represent our t for our analytical equation
        iNewA = numeratorA/(1+eConstant*math.e**(-exponent*timeCounter))
        # Make a new variable potential that takes the absolute value of Euler-Analytical
        potential = abs(iNewE-iNewA)
        # If the potential is higher than the currMax set the currMax to the potential
        if(potential>currMax):
            currMax = potential
        # Set sPrev and iPrev to this iterations numbers to prepare for the next iteration
        sPrev = sNewE
        iPrev = iNewE
    # After the time loop has concluded return the max
    return currMax





