# This code simulates the idea for a blueberry bank transition line - Simulation

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plane 1 parameters (main bank slope data fetched from plane_estimation.py)
a1 = 0.054858637984286356
b1 = 0.8536365064458198
c1 = 0.5179722431766831
d1 = -0.41266453699801897

# Grid for plane 1
x1 = np.linspace(-10, 10, 20)
y1 = np.linspace(-10, 0, 20)
x1, y1 = np.meshgrid(x1, y1)
z1 = (-d1 - a1*x1 - b1*y1) / c1 # Plane 1 equation

# Plane 2 parameters (flat bank slope data fetched from plane_estimation.py)
a2 = -0.1342691281643828
b2 = 0.27132129599290317
c2 = 0.9530774132056157
d2 = -0.6267725595793305

# Grid for plane 2
x2 = np.linspace(-10, 10, 20)
y2 = np.linspace(0, 10, 20)
x2, y2 = np.meshgrid(x2, y2)
z2 = (-d2 - a2*x1 - b2*y1) / c2 # Plane 2 equation

"""
# Transition line - Simulated brute force 
transition_line_x = x1[-1, :]    
transition_line_y = y1[-1, :]    
transition_line_z = z1[-1, :]    
"""

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot slope plane
ax.plot_surface(x1, y1, z1, alpha=0.5, rstride=1, cstride=1, color='cyan', edgecolor='k')

# Plot flat top plane
ax.plot_surface(x2, y2, z2, alpha=0.7, rstride=1, cstride=1, color='lime', edgecolor='k')

# Plot transition line
#ax.plot(transition_line_x, transition_line_y, transition_line_z, color='red', linewidth=3, label='Transition Line')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Blueberry bank plane estimation')
ax.legend()

plt.show()
