# This program is to select significant KW evetns
import json
import numpy as np
import netCDF4 as nc

from typing import Tuple
from itertools import product
from matplotlib import pyplot as plt


def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
    fpath: str = "/data92/b11209013/MPAS/merged_data/"

    exp_list: list[str] = ["CNTL", "NCRF"]

    var_list: list[str] = ["q1"]

    iter_list: list[str, Tuple[str, str]] = list(product(var_list, exp_list))

    # load dimensions
    with nc.Dataset(f"{fpath}{exp_list[0]}/{var_list[0]}.nc", "r") as ds:
        dims: dict[str, np.ndarray] = {
            key: ds[key][:] for key in ds.dimensions.keys()
        }

        lat_lim: list[int] = np.where(
            (dims["lat"] >= -5.0) &
            (dims["lat"] <= 5.0)
        )[0]

        dims["lat"] = dims["lat"][lat_lim]
        dims["time"] = dims["time"][-360:]

    # load data
    data: dict[str, np.ndarray] = {var: {} for var in var_list}

    for (var, exp) in iter_list:
        with nc.Dataset(f"{fpath}{exp}/{var}.nc", "r") as ds:
            data[var][exp] = ds[var][-360:][..., lat_lim, :].mean(axis=2)

    # load event json
    with open("/home/b11209013/2025_Research/AOGS/File/events.json", "r") as f:
        event: dict[str, np.ndarray] = json.load(f)

    # ==== 2. Processing data ==== #
    # Compute vertically integral of Q1
    def vert_int(data, lev):
        data_ave = (data[:, 1:] + data[:, :-1]) / 2.0
        data_vint = -np.sum(data_ave * np.diff(lev*100.0)
                            [None, :, None], axis=1)

        return data_vint * 86400.0 / 9.81 / 2.5e6

    q1_vint: dict[str, np.ndarray] = {
        exp: vert_int(data["q1"][exp], dims["lev"])
        for exp in exp_list
    }

    # compute anomalies of Q1
    anom: dict[str, np.ndarray] = {
        exp: q1_vint[exp] - q1_vint[exp].mean(axis=(0, -1), keepdims=True)
        for exp in exp_list
    }

    # ==== 3. Select events ==== #
    shifting: dict[str, np.ndarray] = {
        exp: np.array([
            np.roll(anom[exp][:, x], anom[exp].shape[0]//2-t)
            for (x, t) in zip(event[exp]["active_x"], event[exp]["active_t"])
        ]).mean(axis=0)
        for exp in exp_list
    }

    def find_zeros(data):
        cond = (data[1:] * data[:-1]) <= 0

        dist = np.where(cond)[0] - data.shape[0] // 2

        left_idx = dist[dist < 0][-1] + data.shape[0] // 2 + 1
        right_idx = dist[dist > 0][1] + data.shape[0] // 2 + 1

        return [left_idx, right_idx]

    boundary: dict[str, list[int]] = {
        "CNTL": [-15, 8],
        "NCRF": [-14, 5]
    }

    with open("/home/b11209013/2025_Research/AOGS/File/boundary.json", "w") as f:
        json.dump(convert_numpy(boundary), f, indent=4)

    plt.plot(np.linspace(1, 360, 360) - shifting["NCRF"].shape[0]//2, np.convolve(
        shifting["NCRF"], np.ones(4)/4, mode="same"))
    plt.scatter(np.linspace(1, 360, 360) - shifting["NCRF"].shape[0]//2, np.convolve(
        shifting["NCRF"], np.ones(4)/4, mode="same"), s=2, c="r")
    plt.xlim(-40, 20)
    plt.axhline(0.0, color="k", linestyle="--")
    plt.axvline(-14, color="k", linestyle="--")
    plt.axvline(5, color="k", linestyle="--")
    plt.xlabel("Lag (6 hr)")
    plt.ylabel("Vertically integrated Q1 anomaly (mm/day)")
    plt.show()


if __name__ == "__main__":
    main()
