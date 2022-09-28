import matplotlib.pyplot as plt
import numpy as np
from math import tanh

def f(x, y):
    return -tanh(365/(10**8) * (max(x,0)**0.15) * (100 - max(y,0)) * (450 - 1000)) * (66-0.33*max(y,0)) + 0.3*(max(x,0)**0.15)*(100-max(y,0))

Z =
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');