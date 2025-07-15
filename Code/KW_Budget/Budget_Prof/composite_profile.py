# This program is to composite alpha, Q1, omega
import json
import numpy as np
import netCDF4 as nc

from typing import Tuple
from scipy.ndimage import convolve1d
from itertools import product
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

exec(open("/home/b11209013/Package/Plot_Style.py").read())


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def main():
    # ==== 1. Load data ==== #
    fpath: str = "/data92/b11209013/MPAS/merged_data"

    exp_list: list[str] = ["CNTL", "NCRF"]
    var_list: list[str] = ["u", "w", "q1", "theta"]

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

    # load data
    data: dict[str, dict[str, np.ndarray]] = {
        var: {} for var in var_list
    }

    for (exp, var) in iter_list:
        with nc.Dataset(fpath + f"/{exp}/{var}.nc") as ds:
            data[var][exp] = ds[var][-360:][..., lat_lim, :].mean(axis=2)

    # compute temperature
    data["temp"] = {
        exp: data["theta"][exp] * converter
        for exp in exp_list
    }

    # Compute density
    data["rho"] = {
        exp: dims["lev"][None, :, None]*100.0 /
        287.5 / data["temp"][exp]
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

    # Load events
    with open("/home/b11209013/2025_Research/AOGS/File/events.json", "r") as f:
        events = json.load(f)

    # load boundary
    with open("/home/b11209013/2025_Research/AOGS/File/boundary.json", "r") as f:
        boundary = json.load(f)

    # ==== 2. Processing data ==== #
    # Compute anomalies
    anom: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: data[var][exp] -
            data[var][exp].mean(axis=(0, -1), keepdims=True)
            for exp in exp_list
        } for var in var_list
    }

    # ==== 3. Find east boundary of the KWs ==== #

    # select events
    center_idx = 360//2

    right_bnd = {
        "CNTL": boundary["CNTL"][-1] + 3,
        "NCRF": boundary["NCRF"][-1] + 2
    }

    sel_data: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: np.array([
                np.roll(anom[var][exp][..., x], center_idx-t, axis=0)
                for x, t in zip(events[exp]["active_x"], events[exp]["active_t"])
            ]).mean(axis=0)[center_idx-13:center_idx+15].T
            for exp in exp_list
        }
        for var in var_list
    }

    print(sel_data["q1"]["CNTL"].shape)

    def find_left_bnd(data):
        diff_prof = data[:, -20:-6] - data[:, -1][:, None]

        l2_norm = np.mean(np.abs(diff_prof)**2, axis=0)

        left_bnd = np.argmin(l2_norm) + 160

        return left_bnd

    left_bnd = {
        exp: -13
        for exp in exp_list
    }

    right_bnd = {
        exp: 15
        for exp in exp_list
    }

    # ==== 3. Plot out profiles ==== #
    x = (np.arange(left_bnd["CNTL"],
                   right_bnd["CNTL"], 1)/4.0)
    z = dims["lev"]

    xx, zz = np.meshgrid(x, z)
    apply_custom_plot_style()

    plt.figure(figsize=(32, 14))

    p = plt.pcolormesh(x, z,
                       sel_data["q1"]["CNTL"]*86400.0/1004.5,
                       norm=TwoSlopeNorm(0.0, vmin=-2, vmax=10),
                       cmap="RdBu_r")
    c = plt.contour(x, z,
                    sel_data["alpha"]["CNTL"], colors="k",
                    levels=np.arange(-0.04, 0.04, 0.002))
    plt.quiver(xx[::2, ::2], zz[::2, ::2],
               sel_data["u"]["CNTL"][::2, ::2],
               sel_data["w"]["CNTL"][::2, ::2]*400,
               scale=300, width=2e-3
               )
    plt.subplots_adjust(left=0.1, right=1.05, top=0.9, bottom=0.1)
    plt.gca().invert_xaxis()
    plt.yscale("log")
    plt.xticks(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7).astype(int))
    plt.yticks([200, 300, 400, 500, 600, 800, 1000],
               [200, 300, 400, 500, 600, 800, 1000])
    plt.ylim(1000, 175)
    plt.colorbar(p, label=r"$Q_1$ [ K/day ]")
    plt.clabel(c, inline=True, fontsize=16)
    plt.xlabel("Lag Time [ day ]")
    plt.ylabel("Pressure [ hPa ]")
    plt.title("CNTL")
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/CNTL_comp.png", dpi=500)
    plt.show()

    x = (np.arange(left_bnd["NCRF"],
                   right_bnd["NCRF"], 1)/4.0)
    z = dims["lev"]

    xx, zz = np.meshgrid(x, z)

    plt.figure(figsize=(32, 14))

    p = plt.pcolormesh(x, z,
                       sel_data["q1"]["NCRF"]*86400.0/1004.5,
                       norm=TwoSlopeNorm(0.0, vmin=-2, vmax=10),
                       cmap="RdBu_r")
    c = plt.contour(x, z,
                    sel_data["alpha"]["NCRF"], colors="k",
                    levels=np.arange(-0.04, 0.04, 0.002))
    plt.quiver(xx[::2, ::2], zz[::2, ::2],
               sel_data["u"]["NCRF"][::2, ::2],
               sel_data["w"]["NCRF"][::2, ::2]*400,
               scale=300, width=2e-3
               )
    plt.subplots_adjust(left=0.1, right=1.05, top=0.9, bottom=0.1)
    plt.gca().invert_xaxis()
    plt.yscale("log")
    plt.xticks(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7).astype(int))
    plt.yticks([200, 300, 400, 500, 600, 800, 1000],
               [200, 300, 400, 500, 600, 800, 1000])
    plt.ylim(1000, 175)
    plt.colorbar(p, label=r"$Q_1$ [ K/day ]")
    plt.clabel(c, inline=True, fontsize=16)
    plt.xlabel("Lag Time [ day ]")
    plt.ylabel("Pressure [ hPa ]")
    plt.title("NCRF")
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/NCRF_comp.png", dpi=500)
    plt.show()

    with open("/home/b11209013/2025_Research/AOGS/File/boundary.json", "w") as f:
        json.dump(convert_numpy({
            "CNTL": [left_bnd["CNTL"], right_bnd["CNTL"]],
            "NCRF": [left_bnd["NCRF"], right_bnd["NCRF"]]
        }), f, indent=4)


if __name__ == '__main__':
    main()
