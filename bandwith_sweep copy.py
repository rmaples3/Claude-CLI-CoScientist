import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Optimized Parameters
lambda_c = 0.0405
M = 0.0024
t1 = -0.1030
t2 = 1.2186
t3 = -0.0052

# 2. Brillouin Zone Mesh
k = np.linspace(-np.pi, np.pi, 200)
kx, ky = np.meshgrid(k, k)

# 3. Hamiltonian Components
dx = np.sin(kx) + lambda_c * np.sin(2 * kx)
dy = np.sin(ky) + lambda_c * np.sin(2 * ky)
dz = (M 
      + t1 * (np.cos(kx) + np.cos(ky)) 
      + t2 * (np.cos(kx) * np.cos(ky)) 
      + t3 * (np.cos(2*kx) + np.cos(2*ky)))

# Lower and Upper Bands
E_minus = -np.sqrt(dx**2 + dy**2 + dz**2)
E_plus = np.sqrt(dx**2 + dy**2 + dz**2)

# 4. Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot both bands to show the isolated topological gap
surf1 = ax.plot_surface(kx, ky, E_minus, cmap='viridis', edgecolor='none', alpha=0.9)
surf2 = ax.plot_surface(kx, ky, E_plus, cmap='plasma', edgecolor='none', alpha=0.3)

ax.set_title(r"Optimized $C=-2$ Band Structure (Flatness Ratio $\approx 4.87$)", fontsize=14)
ax.set_xlabel(r'$k_x$', fontsize=12)
ax.set_ylabel(r'$k_y$', fontsize=12)
ax.set_zlabel(r'$E(\mathbf{k}) / t$', fontsize=12)

# Set strict Z-limits to emphasize the gap vs bandwidth
ax.set_zlim(-3, 3)

plt.tight_layout()
plt.show()