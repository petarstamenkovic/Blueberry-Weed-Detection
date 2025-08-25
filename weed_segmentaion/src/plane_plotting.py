# This code simulates the idea for a blueberry bank - Detected two slopes and draw them with Carthesian parameters

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plane 1 parameters (main bank slope data fetched from plane_estimation.py)
a1 = 0.056837902614027935
b1 = 0.1791423314460837
c1 = 0.982179962079506
d1 = -0.5602395956705073

# Grid for plane 1
x1 = np.linspace(-10, 10, 20)
y1 = np.linspace(-10, 0, 20)
x1, y1 = np.meshgrid(x1, y1)
z1 = (-d1 - a1*x1 - b1*y1) / c1 # Plane 1 equation

# Plane 2 parameters (flat bank slope data fetched from plane_estimation.py)
a2 = 0.026610171452929748
b2 = 0.8292732018703666
c2 = 0.5582095085493582
d2 = -0.3958627341483104

# Grid for plane 2
x2 = np.linspace(-10, 10, 20)
y2 = np.linspace(0, 10, 20)
x2, y2 = np.meshgrid(x2, y2)
z2 = (-d2 - a2*x2 - b2*y2) / c2 # Plane 2 equation

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot slope plane
ax.plot_surface(x1, y1, z1, alpha=0.5, rstride=1, cstride=1, color='cyan', edgecolor='k')

# Plot flat top plane
ax.plot_surface(x2, y2, z2, alpha=0.7, rstride=1, cstride=1, color='lime', edgecolor='k')


# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Blueberry bank plane estimation')
ax.legend()

plt.show()
