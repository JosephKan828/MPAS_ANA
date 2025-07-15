# This program is to composite alpha, Q1, omega
import json
import numpy as np
import netCDF4 as nc

from typing import Tuple
from scipy.ndimage import convolve1d
from itertools import product
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

exec(open("/home/b11209013/Package/Plot_Style.py").read())


def compute_stability(
        lev: np.ndarray,
        theta: np.ndarray,
        density: np.ndarray,
) -> np.ndarray:

    theta_mean = theta.mean(axis=-1, keepdims=True)

    density_mean = density.mean(axis=-1, keepdims=True)

    theta_grad = np.gradient(theta_mean, lev*100.0, axis=0)

    return -theta_grad / (density_mean * theta_mean)


def main():
    # ==== 1. Load data ==== #
    fpath: str = "/data92/b11209013/MPAS/merged_data"

    exp_list: list[str] = ["CNTL", "NCRF"]
    var_list: list[str] = ["w", "q1", "theta"]

    iter_list: list[Tuple] = list(product(exp_list, var_list))

    # load dimensions
    with nc.Dataset(fpath + "/CNTL/theta.nc") as ds:
        dims: dict[str, np.ndarray] = {
            key: ds[key][:]
            for key in ds.dimensions.keys()
        }

        lat_lim: np.ndarray = np.where(
            (dims["lat"] >= -5.0) & (dims["lat"] <= 5.0))[0]

        dims["lat"] = dims["lat"][lat_lim]
        dims["time"] = dims["time"][-360:]

        converter: np.ndarray = (
            1000.0 / dims["lev"][None, :, None]) ** (-0.286)

    print("Finished: Loading dimensions")

    # load data
    data: dict[str, dict[str, np.ndarray]] = {
        var: {} for var in var_list
    }

    for (exp, var) in iter_list:
        with nc.Dataset(fpath + f"/{exp}/{var}.nc") as ds:
            data[var][exp] = ds[var][-360:][..., lat_lim, :].mean(axis=2)

        print(f"Finished: Loading {exp} {var}")

    print("Finished: Loading data")

    # Compute temperature
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

    var_list = data.keys()

    print("Finished: Loading data")

    # Load events
    with open("/home/b11209013/2025_Research/AOGS/File/events.json", "r") as f:
        events = json.load(f)

    # load boundary
    with open("/home/b11209013/2025_Research/AOGS/File/boundary.json", "r") as f:
        bnd = json.load(f)

    # ==== 2. Processing data ==== #
    # Compute anomalies
    anom: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: data[var][exp] -
            data[var][exp].mean(axis=(0, -1), keepdims=True)
            for exp in exp_list
        } for var in var_list
    }

    print("Finished: Computing anomalies")

    # ==== 3. Find east boundary of the KWs ==== #

    # select events
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
            sel_data["theta"][exp],
            sel_data["rho"][exp]
        ) for exp in exp_list
    }

    print("Finished: Computing stability")

    # ==== 4. Compute budget ==== #
    # Compute generation
    def compute_generation(
            lev: np.ndarray,
            stab: np.ndarray,
            alpha: np.ndarray,
            heating: np.ndarray
    ) -> np.ndarray:
        return 287.5 * heating * alpha / (lev[:, None]*100.0 * 1004.5 * stab)

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
    def compute_conversion(
        alpha: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        return alpha * omega

    conv: dict[str, np.ndarray] = {
        exp: compute_conversion(
            sel_data["alpha"][exp],
            sel_data["omega"][exp]
        )
        for exp in exp_list
    }

    print("Finished: Computing conversion")

    # Compute specific volume variance
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

    # ==== 5. plot out profiles ==== #
    apply_custom_plot_style()

    x = np.arange(bnd["CNTL"][0], bnd["CNTL"][-1], 1) / 4.0
    z = dims["lev"]

    xx, zz = np.meshgrid(x, z)

    # plot out the profiles
    ## Plot Generation
    fig = plt.figure(figsize=(26, 15))

    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[20, 1], hspace=0.05,
        width_ratios=[9, 1], wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    c = ax1.pcolormesh(
        xx, zz,
        gen["CNTL"],
        cmap="PiYG_r", norm=TwoSlopeNorm(0.0),
    )
    ax1.set_yscale("log")
    ax1.set_yticks([200, 300, 400, 500, 600, 800, 1000], [200, 300, 400, 500, 600, 800, 1000])
    ax1.set_xlim(x[-1], x[0])
    ax1.set_ylim(1000, 175)
    ax1.set_title(r"CNTL $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime$")

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(gen["CNTL"].sum(axis=1), z)
    ax2.set_ylim(1000, 175)
    ax_cb = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(
        c, cax=ax_cb, orientation='horizontal',
        label="Generation [ $J\;kg^{-1}\;s^{-1}$ ]"
        )
    
    bbox_main = ax1.get_position()
    bbox_cb = ax_cb.get_position()
    dy = bbox_main.y0 - bbox_cb.y0
    ax_cb.set_position([bbox_cb.x0, bbox_cb.y0 - dy, bbox_cb.width, bbox_cb.height])

    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Generation/CNTL.png", dpi=300)
    plt.show()
    plt.close(fig)


    fig = plt.figure(figsize=(26, 15))

    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[20, 1], hspace=0.05,
        width_ratios=[9, 1], wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    c = ax1.pcolormesh(
        xx, zz,
        gen["NCRF"],
        cmap="PiYG_r", norm=TwoSlopeNorm(0.0),
    )
    ax1.set_yscale("log")
    ax1.set_yticks([200, 300, 400, 500, 600, 800, 1000], [200, 300, 400, 500, 600, 800, 1000])
    ax1.set_xlim(x[-1], x[0])
    ax1.set_ylim(1000, 175)
    ax1.set_title(r"NCRF $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime$")

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(gen["NCRF"].sum(axis=1), z)
    ax2.set_ylim(1000, 175)
    ax_cb = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(
        c, cax=ax_cb, orientation='horizontal',
        label="Generation [ $J\;kg^{-1}\;s^{-1}$ ]"
        )
    
    bbox_main = ax1.get_position()
    bbox_cb = ax_cb.get_position()
    dy = bbox_main.y0 - bbox_cb.y0
    ax_cb.set_position([bbox_cb.x0, bbox_cb.y0 - dy, bbox_cb.width, bbox_cb.height])

    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Generation/NCRF.png", dpi=300)
    plt.show()
    plt.close(fig)

    # Plot conversion
    fig = plt.figure(figsize=(26, 15))

    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[20, 1], hspace=0.05,
        width_ratios=[9, 1], wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    c = ax1.pcolormesh(
        xx, zz,
        conv["CNTL"],
        cmap="PiYG_r", norm=TwoSlopeNorm(0.0),
    )
    ax1.set_yscale("log")
    ax1.set_yticks([200, 300, 400, 500, 600, 800, 1000], [200, 300, 400, 500, 600, 800, 1000])
    ax1.set_xlim(x[-1], x[0])
    ax1.set_ylim(1000, 175)
    ax1.set_title(r"CNTL $\alpha^\prime Q^\prime$")

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(conv["CNTL"].sum(axis=1), z)
    ax2.set_ylim(1000, 175)
    ax_cb = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(
        c, cax=ax_cb, orientation='horizontal',
        label="Conversion [ $J\;kg^{-1}\;s^{-1}$ ]"
        )
    cb.set_ticks([-5e-7, -3e-7, 0, 1e-7, 2e-7])
    
    bbox_main = ax1.get_position()
    bbox_cb = ax_cb.get_position()
    dy = bbox_main.y0 - bbox_cb.y0
    ax_cb.set_position([bbox_cb.x0, bbox_cb.y0 - dy, bbox_cb.width, bbox_cb.height])

    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Generation/CNTL.png", dpi=300)
    plt.show()
    plt.close(fig)

    
    fig = plt.figure(figsize=(26, 15))

    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[20, 1], hspace=0.05,
        width_ratios=[9, 1], wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    c = ax1.pcolormesh(
        xx, zz,
        gen["NCRF"],
        cmap="PiYG_r", norm=TwoSlopeNorm(0.0),
    )
    ax1.set_yscale("log")
    ax1.set_yticks([200, 300, 400, 500, 600, 800, 1000], [200, 300, 400, 500, 600, 800, 1000])
    ax1.set_xlim(x[-1], x[0])
    ax1.set_ylim(1000, 175)
    ax1.set_title(r"NCRF $\frac{R}{p C_p \sigma} \alpha^\prime Q^\prime$")

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(gen["NCRF"].sum(axis=1), z)
    ax2.set_ylim(1000, 175)
    ax_cb = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(
        c, cax=ax_cb, orientation='horizontal',
        label="Generation [ $J\;kg^{-1}\;s^{-1}$ ]"
        )
    
    bbox_main = ax1.get_position()
    bbox_cb = ax_cb.get_position()
    dy = bbox_main.y0 - bbox_cb.y0
    ax_cb.set_position([bbox_cb.x0, bbox_cb.y0 - dy, bbox_cb.width, bbox_cb.height])

    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EAPE/Generation/NCRF.png", dpi=300)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
