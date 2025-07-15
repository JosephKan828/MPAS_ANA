# This program is to compute cross spectrum of KW
import numpy as np
import netCDF4 as nc

from einops import repeat
from itertools import product
from joblib import Parallel, delayed
from scipy.signal import detrend
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# compute stability
def compute_stability(
    lev: np.ndarray,
    rho: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:

    mean_theta = theta.mean(axis=-1, keepdims=True)

    mean_rho = rho.mean(axis=-1, keepdims=True)

    theta_grad = np.gradient(theta, lev*100.0, axis=1)

    return -theta_grad / (mean_rho * mean_theta)


def compute_cross(
    alpha: np.ndarray,
    heating: np.ndarray
) -> np.ndarray:

    Nt, Nx = alpha.shape[0], alpha.shape[-1]

    a_fft = np.fft.fft(alpha, axis=0)
    a_fft = np.fft.ifft(a_fft, axis=-1)

    h_fft = np.fft.fft(heating, axis=0)
    h_fft = np.fft.ifft(h_fft, axis=-1)

    ps = (a_fft * h_fft.conj()) / (Nt * Nx)**2.0

    kernel = np.array([1.0, 2.0, 1.0]) / 4.0

    ps_conv = convolve1d(ps, kernel, axis=0, mode="reflect")
    ps_conv = convolve1d(ps_conv, kernel, axis=-1, mode="reflect")

    return ps_conv


def compute_gen(
    lev: np.ndarray,
    stab: np.ndarray,
    alpha: np.ndarray,
    heating: np.ndarray,
) -> np.ndarray:
    lev_bc = lev[None, None, :, None, None]*100.0

    coeff = 287.5 / (lev_bc * 1004.5 * stab)

    gen = np.array(Parallel(n_jobs=8)(
        delayed(compute_cross)(
            coeff[i]*alpha[i], heating[i]
        )
        for i in range(alpha.shape[0])
    )).mean(axis=0)

    return gen.real


def vert_int(
    lev: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:

    return -np.trapz(data, lev*100.0, axis=1) / 9.81


def main():
    # ==== 1. load data ==== #
    fpath: str = "/data92/b11209013/MPAS/merged_data/"

    exp_list: list[str] = ["CNTL", "NCRF"]

    var_list: list[str] = ["w", "q1", "theta"]

    iter_list: list[str] = list(product(var_list, exp_list))

    # Load dimensions
    with nc.Dataset(f"{fpath}{exp_list[0]}/{var_list[0]}.nc", "r") as ds:
        dims: dict[str, np.ndarray] = {
            key: ds[key][...]
            for key in ds.dimensions.keys()
            if not key == "time"
        }

    lat_lim: list[int] = np.where(
        (dims["lat"] >= -5.0) & (dims["lat"] <= 5.0)
    )[0]

    dims["lat"] = dims["lat"][lat_lim]

    lev_bc = repeat(
        dims["lev"],
        "lev -> time lev lat lon",
        time=360, lat=dims["lat"].size, lon=dims["lon"].size
    )

    converter: np.ndarray = (1000.0 / lev_bc) ** (-0.286)

    # Load data
    data: dict[str, dict[str, np.ndarray]] = {var: {} for var in var_list}

    for (var, exp) in iter_list:

        with nc.Dataset(f"{fpath}{exp}/{var}.nc", "r") as ds:

            if var == "theta":
                data[var][exp] = ds[var][-360:][..., lat_lim, :]

            else:
                data[var][exp] = ds[var][-360:][..., lat_lim, :]

    # Com"pute temperature
    data["t"] = {
        exp: data["theta"][exp] * converter
        for exp in exp_list
    }

    # Compute density
    data["rho"] = {
        exp: lev_bc*100.0 / (287.5 * data["t"][exp])
        for exp in exp_list
    }

    # Compute specific volumn
    data["alpha"] = {
        exp: 1 / data["rho"][exp]
        for exp in exp_list
    }

    # Compute omega
    data["omega"] = {
        exp: - 9.81 * data["rho"][exp] * data["w"][exp]
        for exp in exp_list
    }

    var_list = data.keys()

    print("Finshed: Loading data")

    # ==== 2. Proces data ==== #
    # compute anomalies
    data_anom: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: data[var][exp] -
            data[var][exp].mean(axis=(0, -1), keepdims=True)
            for exp in exp_list
        }
        for var in var_list
    }

    print("Finished: Computing anomalies")

    # windowing data
    hanning = np.hanning(120)[:, None, None, None]
    data_wind: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: np.array([detrend(
                data_anom[var][exp][i*60:i*60+120] * hanning,
                axis=0
            )
                for i in range(5)
            ])
            for exp in exp_list
        }
        for var in var_list
    }

    # compute stability
    stab: dict[str, np.ndarray] = {
        exp: compute_stability(
            dims["lev"],
            data["rho"][exp],
            data["theta"][exp]
        )
        for exp in exp_list
    }

    stab_wind: dict[str, np.ndarray] = {
        exp: np.array([
            stab[exp][i*60:i*60+120]
            for i in range(5)
        ])
        for exp in exp_list
    }

    print("Finished: Compute compute_stability")

    # ==== 3. Compute generation ==== #
    # generation
    gen: dict[str, np.array] = {
        exp: vert_int(dims["lev"], compute_gen(
            dims["lev"], stab_wind[exp],
            data_wind["alpha"][exp], data_wind["q1"][exp]
        ))
        for exp in exp_list
    }

    print(gen["CNTL"].sum())

    wn = np.fft.fftshift(np.fft.fftfreq(
        dims["lon"].size, d=1/dims["lon"].size))
    fr = np.fft.fftshift(np.fft.fftfreq(120, d=1/4))

    plt.pcolormesh(wn, fr, np.fft.fftshift(gen["CNTL"].sum(axis=1)))
    plt.xlim(-15, 15)
    plt.ylim(0, 0.5)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
