# This program is to show the SST profile
import numpy as np
from matplotlib import pyplot as plt

sin_lat = np.linspace(-1, 1, 360)

lat = np.arcsin(sin_lat)

sst = 27.0 * (1 - 0.5 * np.sin(3*lat/2)**2 - 0.5 * np.sin(3*lat/2)**4)

sst[np.abs(lat)>np.pi/3] = 0.0

plt.rcParams['font.family'] = "serif"
axes = plt.gca()

plt.plot(np.rad2deg(lat), sst, color="k", linewidth=2)
axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)
plt.xticks(np.linspace(-90, 90, 7))
plt.yticks(np.linspace(0, 25, 6))
plt.xlim(-90, 90)
plt.ylim(0, 30)
plt.xlabel("Latitude (degrees)")
plt.ylabel("SST (Celsius)")
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/SST_prof.png")
plt.show()
plt.close()
