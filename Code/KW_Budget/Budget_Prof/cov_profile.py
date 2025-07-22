import json
import numpy as np
import netCDF4 as nc

from typing import Tuple
from scipy.ndimage import convolve1d
from itertools import product
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend explicitly

exec(open("/home/b11209013/Package/Plot_Style.py").read())

# Compute stability
def compute_stability(
    lev: np.ndarray,
    theta: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:

    theta_mean = theta.mean(axis=(0, -1), keepdims=True)

    density_mean = density.mean(axis=(0, -1), keepdims=True)

    theta_grad = np.gradient(theta_mean, lev * 100.0, axis=1)

    return (-theta_grad / (density_mean * theta_mean)).squeeze()[:, None]


# compute EAPE generation
def compute_generation(
    lev: np.ndarray, stab: np.ndarray, alpha: np.ndarray, heating: np.ndarray
) -> np.ndarray:
    return 287.5 * heating * alpha / (lev[:, None] * 100.0 * 1004.5 * stab)


# compute EAPE conversion
def compute_conversion(
    alpha: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    return alpha * omega

# compute vertical integral
def vert_int(
        lev: np.ndarray,
        data: np.ndarray,
) -> np.float64:
    print(data.shape)
    return -np.trapz(data, lev*100.0) / -np.trapz(np.ones_like(lev), lev*100.0)

# Load dimensions
fpath: str = "/data92/b11209013/MPAS/merged_data"

exp_list: list[str] = ["CNTL", "NCRF"]

var_list: list[str] = ["w", "q1", "theta"]

iter_list: list[Tuple] = list(product(exp_list, var_list))

# load dimensions
with nc.Dataset(fpath + "/CNTL/theta.nc") as ds:

    dims: dict[str, np.ndarray] = {key: ds[key][:] for key in ds.dimensions.keys()}

    lat_lim: np.ndarray = np.where((dims["lat"] >= -5.0) & (dims["lat"] <= 5.0))[0]

    dims["lat"] = dims["lat"][lat_lim]

    dims["time"] = dims["time"][-360:]
    print(dims["lev"])
    converter: np.ndarray = (1000.0 / dims["lev"][None, :, None]) ** (-0.286)

print("Finished: Loading dimensions")

# Load variables
data: dict[str, dict[str, np.ndarray]] = {var: {} for var in var_list}

for exp, var in iter_list:
    with nc.Dataset(fpath + f"/{exp}/{var}.nc") as ds:
        data[var][exp] = ds[var][-360:][..., lat_lim, :].mean(axis=2)

    print(f"Finished: Loading {exp} {var}")

print("Finished: Loading data")

# Compute specific variables
data["temp"] = {
    exp: data["theta"][exp] * converter
    for exp in exp_list
}

# Compute density
data["rho"] = {
    exp: dims["lev"][None, :, None]*100.0 / 287.5 / data["temp"][exp]
    for exp in exp_list
}

# compute omega
data["omega"] = {
    exp: -9.81 * data["rho"][exp] * data["w"][exp]
    for exp in exp_list
}

# compute alpha
data["alpha"] = {
    exp: 1 / data["rho"][exp]
    for exp in exp_list
}
var_list = list(data.keys())
print("Finished: Compute variables")

# Load events
with open("/home/b11209013/2025_Research/AOGS/File/events.json", "r") as f:
    events = json.load(f)

# load boundary
with open("/home/b11209013/2025_Research/AOGS/File/boundary.json", "r") as f:
    bnd = json.load(f)

anom: dict[str, dict[str, np.ndarray]] = {
    var: {
        exp: data[var][exp] -
        data[var][exp].mean(axis=(0, -1), keepdims=True)
        for exp in exp_list
    } for var in var_list
}

center_idx = 360//2

sel_data: dict[str, dict[str, np.ndarray]] = {
    var: {
        exp: np.array([
            np.roll(anom[var][exp][..., x], center_idx-t, axis=0)
            for x, t in zip(events[exp]["active_x"], events[exp]["active_t"])
        ]).mean(axis=0)[center_idx+bnd[exp][0]:center_idx+bnd[exp][-1]].T
        for exp in exp_list
    }
    for var in var_list
}

print("Finished: Selecting data")

# Compute stability
stab: dict[str, np.ndarray] = {
    exp: compute_stability(
        dims["lev"],
        data["theta"][exp],
        data["rho"][exp]
    ) for exp in exp_list
}

print("Finished: Computing stability")

# compute generation
gen: dict[str, np.ndarray] = {
    exp: compute_generation(
        dims["lev"],
        stab[exp],
        sel_data["alpha"][exp],
        sel_data["q1"][exp]
    ) for exp in exp_list
}

print("Finished: Computing generation")

# compute conversion
conv: dict[str, np.ndarray] = {
    exp: compute_conversion(
        sel_data["alpha"][exp],
        sel_data["omega"][exp]
    )
    for exp in exp_list
}

print("Finished: Computing conversion")

# compute specific volume variance
a_var: dict[str, np.ndarray] = {
    exp: sel_data["alpha"][exp] * sel_data["alpha"][exp]
    for exp in exp_list
}

# Compute variance tendency
var_tend: dict[str, np.ndarray] = {
    exp: np.gradient(a_var[exp] / (2*stab[exp]), 6*3600.0, axis=1)
    for exp in exp_list
}

print("Finished: Computing variance tendency")

# Compute difference in generation
gen_diff: np.ndarray = gen["NCRF"] - gen["CNTL"]

# Compute difference in conversion
conv_diff: np.ndarray = conv["NCRF"] - conv["CNTL"]

# Compute difference in variance
var_diff: np.ndarray = a_var["NCRF"] - a_var["CNTL"]

# Compute difference in tendency
tend_diff: np.ndarray = var_tend["NCRF"] - var_tend["CNTL"]

apply_custom_plot_style() # type: ignore

# setup dimensions
x = np.arange(bnd["CNTL"][0], bnd["CNTL"][-1], 1) / 4.0
z = dims["lev"]

z_lim = np.argmin(np.abs(z-150))+1
z = z[:z_lim]

xx, zz = np.meshgrid(x, z[:z_lim])

# plot figure
fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    gen["CNTL"][:z_lim],
    cmap="PiYG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"CNTL $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime$")
cb = plt.colorbar(c, label=r"Generation [ $J kg^{-1} s^{-1}$ ]")
cb.set_ticks([-0.001, 0, 0.001])

plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Generation/CNTL.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    gen["NCRF"][:z_lim],
    cmap="PiYG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"NCRF $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime$")
cb = plt.colorbar(c, label=r"Generation [ $J kg^{-1} s^{-1}$ ]")
cb.set_ticks([-0.001, 0, 0.001])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Generation/NCRF.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    (gen["NCRF"] - gen["CNTL"])[:z_lim],
    cmap="PiYG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
ct = plt.contour(
    xx, zz,
    gen["CNTL"][:z_lim],
    colors="k", linewidths=2,
    levels=[-0.0006, -0.0003, 0.0003, 0.0006]
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"Difference $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime$")
plt.clabel(ct, inline=1, fontsize=12)
cb = plt.colorbar(c, label=r"Generation [ $J kg^{-1} s^{-1}$ ]")
cb.set_ticks([-0.001, 0, 0.001])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Generation/diff.png",)
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    conv["CNTL"][:z_lim],
    cmap="BrBG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"CNTL $\alpha^\prime \omega^\prime$")
cb = plt.colorbar(c, label=r"Conversion [ $J kg^{-1} s^{-1}$ ]")
cb.set_ticks([-0.001, 0, 0.001])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Conversion/CNTL.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    conv["NCRF"][:z_lim],
    cmap="BrBG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"NCRF $\alpha^\prime \omega^\prime$")
cb = plt.colorbar(c, label=r"Conversion [ $J kg^{-1} s^{-1}$ ]")
cb.set_ticks([-0.001, 0, 0.001])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Conversion/NCRF.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    (conv["NCRF"] - conv["CNTL"])[:z_lim],
    cmap="BrBG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
ct = plt.contour(
    xx, zz,
    conv["CNTL"][:z_lim],
    colors="k", linewidths=2,
    levels=[-0.0006, -0.0003, 0.0003, 0.0006]
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"Difference $\alpha^\prime \omega^\prime$")
plt.clabel(ct, inline=1, fontsize=12)
cb = plt.colorbar(c, label=r"Conversion[ $J kg^{-1} s^{-1}$ ]")
cb.set_ticks([-0.001, 0, 0.001])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Conversion/diff.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    a_var["CNTL"][:z_lim],
    cmap="Blues", vmin=0, vmax=0.00012
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"CNTL $\alpha^{\prime 2}$")
cb = plt.colorbar(c, label=r"Variance [ $kg^2 m^{-6}$ ]")
cb.set_ticks([0, 0.00005, 0.0001])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Variance/CNTL.png")
plt.close(fig)


fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    a_var["NCRF"][:z_lim],
    cmap="Blues", vmin=0, vmax=0.00012
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"NCRF $\alpha^{\prime 2}$")
cb = plt.colorbar(c, label=r"Variance [ $kg^2 m^{-6}$ ]")
cb.set_ticks([0, 0.00005, 0.0001])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Variance/NCRF.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    (a_var["NCRF"] - a_var["CNTL"])[:z_lim],
    cmap="bwr", vmin=-1e-4, vmax=1e-4
)
ct = plt.contour(
    xx, zz,
    a_var["CNTL"][:z_lim],
    colors="k", linewidths=2,
    levels=[-1e-4, -5e-5, 5e-5, 1e-4]
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"Difference $\alpha^{\prime 2}$")
plt.clabel(ct, inline=1, fontsize=12)
cb = plt.colorbar(c, label=r"Variance [ $m^{6} kg^{-2} $ ]")
cb.set_ticks([-1e-4, 0, 1e-4])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Variance/diff.png")
plt.close(fig)


fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    var_tend["CNTL"][:z_lim],
    cmap="PRGn_r", norm=TwoSlopeNorm(0.0, vmin=-0.0004, vmax=0.0004),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"CNTL $\frac{1}{2\sigma}\alpha^{\prime 2}$")
plt.colorbar(c, label=r"Tendency [ $J kg^{-1} s^{-1}$ ]")
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Tendency/CNTL.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    var_tend["NCRF"][:z_lim],
    cmap="PRGn_r", norm=TwoSlopeNorm(0.0, vmin=-0.0004, vmax=0.0004),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"NCRF $\frac{1}{2\sigma}\alpha^{\prime 2}$")
plt.colorbar(c, label=r"Tendency [ $J kg^{-1} s^{-1}$ ]")
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Tendency/NCRF.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    (var_tend["NCRF"] - var_tend["CNTL"])[:z_lim],
    cmap="PRGn_r", norm=TwoSlopeNorm(0.0, vmin=-0.0004, vmax=0.0004),
)
ct = plt.contour(
    xx, zz,
    var_tend["CNTL"][:z_lim],
    colors="k", linewidths=2,
    levels=[-3e-4, -1e-4, 1e-4, 3e-4]
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"Difference $\frac{\partial}{\partial t}\left(\frac{1}{2\sigma}\alpha^{\prime 2}\right)$")
plt.clabel(ct, inline=1, fontsize=12)
cb = plt.colorbar(c, label=r"Tendency [ $J kg^{-1} s^{-1}$ ]")
cb.set_ticks([-4e-4, -2e-4, 0, 2e-4, 4e-4])
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Tendency/diff.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    (gen["CNTL"] + conv["CNTL"])[:z_lim],
    cmap="BrBG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"CNTL $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime + \alpha^\prime \omega^\prime$")
plt.colorbar(c, label=r"Generation + Conversion [ $J kg^{-1} s^{-1}$ ]")
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Gen+Conv/CNTL.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    (gen["NCRF"] + conv["NCRF"])[:z_lim],
    cmap="BrBG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"NCRF $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime + \alpha^\prime \omega^\prime$")
plt.colorbar(c, label=r"Generation + Conversion [ $J kg^{-1} s^{-1}$ ]")
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Gen+Conv/NCRF.png")
plt.close(fig)

fig = plt.figure(figsize=(16, 9))

c = plt.pcolormesh(
    xx, zz,
    (gen_diff + conv_diff)[:z_lim],
    cmap="BrBG_r", norm=TwoSlopeNorm(0.0, vmin=-0.001, vmax=0.001),
)
plt.yscale("log")
plt.yticks([200, 300, 400, 500, 600, 800, 1000], ["200", "300", "400", "500", "600", "800", "1000"])
plt.xlim(x[-1], x[0])
plt.ylim(1000, 175)
plt.ylabel(r"Pressure [hPa]")
plt.title(r"Difference $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime + \alpha^\prime \omega^\prime$")
plt.colorbar(c, label=r"Generation + Conversion [ $J kg^{-1} s^{-1}$ ]")
plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Gen+Conv/diff.png")
plt.close(fig)