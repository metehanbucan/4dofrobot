import numpy as np
import matplotlib.pyplot as plt

def cubic_trajectory(q0, qf, T, steps=100):
    t = np.linspace(0,T,steps)

    a0 = q0
    a1 = 0
    a2 = 3*(qf - q0)/T**2
    a3 = -2*(qf-q0)/T**3

    q = a0 + a1*t + a2*t**2 + a3*t**3
    return t,q

def cubic_time_scaling(t,T):
    return (3 * (t**2) / (T**2)) - (2 * (t**3) / (T**3))