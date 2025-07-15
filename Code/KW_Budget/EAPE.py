# This program is to compute EAPE generation term in KW life stage
import json
import numpy as np
import netCDF4 as nc

from itertools import product
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def compute_stability(
        lev: np.ndarray,
        theta: np.ndarray,
        density: np.ndarray,
) -> np.ndarray:

    theta_mean = theta.mean(axis=-1, keepdims=True)

    density_mean = density.mean(axis=-1, keepdims=True)

    theta_grad = np.gradient(theta_mean, lev*100.0, axis=1)

    return -theta_grad / (density_mean * theta_mean)


def compute_generation(
    lev: np.ndarray,
    alpha: np.ndarray,
    heating: np.ndarray,
    stability: np.ndarray,
) -> np.ndarray:

    covariance = alpha * heating

    coeff = 287.5 / (lev[None, :, None]*100.0 * 1004.5 * stability)

    return coeff * covariance


def compute_conversion(
    alpha: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:

    return omega * alpha


def vert_int(
    lev: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    return -np.trapz(data, lev*100.0, axis=0)/9.81


def centering_array(data, centering_idx, axis):
    n = data.shape[axis]

    shift = n//2 - centering_idx

    return np.roll(data, shift, axis=axis)


def main():
    # ==== 1. load data ==== #
    fpath: str = "/data92/b11209013/MPAS/merged_data/"

    exp_list: list[str] = ["CNTL", "NCRF"]

    var_list: list[str] = ["theta", "w", "q1"]

    iter_list: list[str] = list(product(var_list, exp_list))

    # load dimension
    with nc.Dataset(fpath + "CNTL/theta.nc", "r") as ds:
        dims = {
            key: ds[key][...]
            for key in ds.dimensions.keys()
        }

    lat_lim: list[int] = np.where(
        (dims["lat"] >= -5.0) & (dims["lat"] <= 5.0))[0]

    dims["lat"] = dims["lat"][lat_lim]

    converter: np.ndarray = (
        1000.0 / dims["lev"][None, :, None]) ** (-0.286)

    print("Finished: Loading dimensions")

    # load data
    data: dict[str, dict[str, np.ndarry]] = {var: {} for var in var_list}

    for (var, exp) in iter_list:
        with nc.Dataset(f"{fpath}{exp}/{var}.nc", "r") as ds:
            data[var][exp] = ds[var][..., lat_lim, :].mean(axis=2)

    data["temp"] = {
        exp: data["theta"][exp] * converter
        for exp in exp_list
    }

    data["rho"] = {
        exp: dims["lev"][None, :, None] *
        100.0 / (287.5 * data["temp"][exp])
        for exp in exp_list
    }

    data["alpha"] = {
        exp: 1/data["rho"][exp]
        for exp in exp_list
    }

    data["omega"] = {
        exp: -9.81 * data["rho"][exp] * data["w"][exp]
        for exp in exp_list
    }

    var_list = data.keys()

    print("Finished: loading data")

    # load KW events
    with open("/home/b11209013/2025_Research/MPAS/File/events.json", "r") as f:
        events: dict[str, dict[str, list]] = json.load(f)

    print("Finished: Loading events")

    # ==== 2. Select field associated with KWs ==== #
    # compute anomalies
    data_anom: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: data[var][exp] -
            data[var][exp].mean(axis=(0, -1), keepdims=True)
            for exp in exp_list
        }
        for var in var_list
    }

    # select data
    tot_sel: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: np.array([
                centering_array(data[var][exp][t], x, axis=-1)
                for (t, x) in zip(
                    events[exp]["active_t"], events[exp]["active_x"]
                )
            ])
            for exp in exp_list
        }
        for var in var_list
    }

    anom_sel: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: np.array([
                centering_array(data_anom[var][exp][t], x, axis=-1)
                for (t, x) in zip(
                    events[exp]["active_t"], events[exp]["active_x"]
                )
            ])
            for exp in exp_list
        }
        for var in var_list
    }

    print("Finished: variables composite")

    # ==== 3. Compute mean and anomalies ==== #
    # Compute statility
    stability: dict[str, np.ndarray] = {
        exp: compute_stability(
            dims["lev"], tot_sel["theta"][exp], tot_sel["rho"][exp])
        for exp in exp_list
    }

    print("Finished: Compute stability")

    # ==== 4. Compute generation and conversion ==== #
    lev_lim = np.argmin(np.abs(dims["lev"]-150.0))

    # Compute conversion
    conversion: dict[str, np.ndarray] = {
        exp: vert_int(dims["lev"][:lev_lim+1], compute_conversion(
            anom_sel["alpha"][exp], anom_sel["omega"][exp]).mean(axis=0)[:lev_lim+1])
        for exp in exp_list
    }

    # Compute generation
    generation: dict[str, np.ndarray] = {
        exp: vert_int(dims["lev"][:lev_lim+1], compute_generation(
            dims["lev"],
            anom_sel["alpha"][exp],
            anom_sel["q1"][exp],
            stability[exp]).mean(axis=0)[:lev_lim+1])
        for exp in exp_list
    }

    plt.pcolormesh(dims["lon"], dims["lev"], anom_sel["q1"]["CNTL"].mean(
        axis=0), cmap="RdBu_r", norm=TwoSlopeNorm(0.0))
    plt.colorbar()
    plt.show()

    print("Finished: Compute generation and conversion")
    """
    xaxis = np.linspace(-50, 50, 199)
    for i in range(1, 6):
        min_lon = -i*10
        max_lon = i*10

        print(f"CNTL generation within ({min_lon}-{max_lon}):\t",
              np.sum(generation["CNTL"][np.abs(xaxis) <= i*10]))
        print(f"NCRF generation within ({min_lon}-{max_lon}):\t",
              np.sum(generation["NCRF"][np.abs(xaxis) <= i*10]))
        print(f"CNTL conversion within ({min_lon}-{max_lon}):\t",
              np.sum(conversion["CNTL"][np.abs(xaxis) <= i*10]))
        print(f"NCRF conversion within ({min_lon}-{max_lon}):\t",
              np.sum(conversion["NCRF"][np.abs(xaxis) <= i*10]))
        print("\t")
    """
    plt.plot(dims["lon"],
             np.convolve(generation["CNTL"], np.ones(33)/33, mode="same"), c="royalblue", label="CNTL gen")
    plt.plot(dims["lon"],
             np.convolve(generation["NCRF"], np.ones(33)/33, mode="same"), c="indianred", label="NCRF gen")
    plt.xticks(np.linspace(-50, 50, 11))
    plt.ylim(-0.6, 0.6)
    plt.grid()
    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/gen_prof.png")
    plt.show()

    plt.plot(dims["lon"],
             np.convolve(conversion["CNTL"], np.ones(33)/33, mode="same"), c="royalblue", label="CNTL gen")
    plt.plot(dims["lon"],
             np.convolve(conversion["NCRF"], np.ones(33)/33, mode="same"), c="indianred", label="NCRF gen")
    plt.xticks(np.linspace(-50, 50, 11))
    plt.ylim(-0.6, 0.6)
    plt.grid()
    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/conv_prof.png")
    plt.show()


if __name__ == '__main__':
    main()
