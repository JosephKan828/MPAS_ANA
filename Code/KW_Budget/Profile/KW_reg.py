# This program is to composite KWs
import json
import numpy as np
import netCDF4 as nc

from itertools import product
from matplotlib import pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend explicitly

exec(open("/home/b11209013/Package/Plot_Style.py").read())

def main():
    # Load dimensions
    fpath: str = "/data92/b11209013/MPAS/merged_data"

    exp_list: list[str] = ["CNTL", "NCRF"]

    var_list: list[str] = ["qc"]

    iter_list: list[tuple] = list(product(exp_list, var_list))

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



    # Load events
    with open("/home/b11209013/2025_Research/AOGS/File/events.json", "r") as f:
        events = json.load(f)

    # load boundary
    with open("/home/b11209013/2025_Research/AOGS/File/boundary.json", "r") as f:
        bnd = json.load(f)
        

    # ==== 2. rolling data ==== #
    center_idx = 360 // 2
    
    data_sel = {var: {} for var in var_list}
    print(bnd["CNTL"])
    for var in var_list:
        for exp in exp_list:
            data_sel[var][exp] = np.array([
                np.roll(data[var][exp][:, :, x], center_idx - t, axis=0)
                for t, x in zip(events[exp]["active_t"], events[exp]["active_x"])
                ]).mean(axis=0).T[:, bnd[exp][0]+center_idx:bnd[exp][1]+1+center_idx]
    
    # ==== 3. plot data ==== #
    apply_custom_plot_style() # type: ignore
    
    plt.figure(figsize=(16, 9))
    p = plt.pcolormesh(
        np.linspace(-3.25, 3.75, 29), dims["lev"],
        data_sel["qc"]["NCRF"] - data_sel["qc"]["CNTL"],
        cmap="RdBu_r", shading="auto"
    )
    ct = plt.contour(
        np.linspace(-3.25, 3.75, 29), dims["lev"],
        data_sel["qc"]["CNTL"], levels=[-6, -3, 3, 6]
        , colors="k", linewidths=2
    )
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.yscale("log")
    plt.xlabel("Lag (days)")
    plt.ylabel("Pressure (hPa)")
    plt.yticks([1000, 800, 600, 400, 300, 200],
                ["1000", "800", "600", "400", "300", "200"])
    plt.ylim(1000, 150)
    plt.title("Convective Heating Difference (NCRF - CNTL)")
    plt.clabel(ct, inline=True, fontsize=12, fmt="%.1f")
    plt.colorbar(p, label=r"$\delta Q_c$ (K/day)")
    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/dqc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Finished: Plotting data")    
    
    
if __name__ == "__main__":
    main()