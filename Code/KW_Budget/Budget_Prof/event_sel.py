# This program is to select significant KW evetns
import json
import numpy as np
import netCDF4 as nc

from typing import Tuple
from itertools import product
from matplotlib import pyplot as plt


def find_local_maximum(data):
    from scipy.ndimage import maximum_filter

    max_filtered = maximum_filter(data, size=(80, 7), mode="reflect")

    local_max_mask = (data == max_filtered) & (
        data >= (data.mean() + 1.96*data.std()))

    coords = np.argwhere(local_max_mask)

    t_len, x_len = data.shape

    coords_valid = [
        (t, x) for t, x, in coords
        if 40 <= t < t_len - 40
    ]

    valid_t, valid_x = [], []
    for i in range(len(coords_valid)):
        valid_t.append(coords_valid[i][0].astype(int))
        valid_x.append(coords_valid[i][1].astype(int))

    return {"active_t": valid_t, "active_x": valid_x}


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

    # ==== 3.Filtering signals ==== #
    # compute FFT2
    def fft2(data: np.ndarray) -> np.ndarray:
        data_fft: np.ndarray = np.fft.fft(data, axis=0)
        data_fft: np.ndarray = np.fft.ifft(data_fft, axis=1)

        return data_fft

    data_fft: dict[str, np.ndarray] = {
        exp: fft2(anom[exp]) for exp in exp_list
    }

    # setup axis
    wn = np.fft.fftfreq(dims["lon"].size, d=1/dims["lon"].size)
    fr = np.fft.fftfreq(dims["time"].size, d=1/4)

    wnm, frm = np.meshgrid(wn, fr)

    def kel_curve(wn, ed): return 86400.0 * \
        np.sqrt(9.81*ed) * wn / (2*np.pi*6.371e6)

    kel_mask = np.where(
        (
            (wnm >= 1) & (wnm <= 15) &
            (frm >= 1/20) & (frm <= 1/2) &
            (frm >= kel_curve(wnm, 10)) & (frm <= kel_curve(wnm, 100))
        ) |
        (
            (wnm <= -1) & (wnm >= -15) &
            (frm <= -1/20) & (frm >= -1/2) &
            (frm <= kel_curve(wnm, 10)) & (frm >= kel_curve(wnm, 100))
        ), 1.0, 0.0
    )

    # reconstruct data
    def ifft2(data: np.ndarray) -> np.ndarray:
        data_ifft: np.ndarray = np.fft.ifft(data, axis=0)
        data_ifft: np.ndarray = np.fft.fft(data_ifft, axis=1)

        return data_ifft

    data_filtered: dict[str, np.ndarray] = {
        exp: ifft2(data_fft[exp] * kel_mask).real
        for exp in exp_list
    }

    # ==== 5. filtering local maximum ==== #
    # select active events
    event_coord: dict[str, np.ndarray] = {
        exp: find_local_maximum(
            data_filtered[exp]
        )
        for exp in exp_list
    }

    # select suppress time
    for exp in exp_list:
        event_coord[exp]["suppress_t"] = []

        for x, t in zip(event_coord[exp]["active_x"], event_coord[exp]["active_t"]):
            event_coord[exp]["suppress_t"].append(
                np.argmin(data_filtered[exp][t:, x]) + t
            )

    # ==== 6. save events ==== #
    with open("/home/b11209013/2025_Research/AOGS/File/events.json", "w") as f:
        json.dump(convert_numpy(event_coord), f, indent=4)

    np.savez("/home/b11209013/2025_Research/AOGS/File/events.npz", **event_coord)


if __name__ == "__main__":
    main()
