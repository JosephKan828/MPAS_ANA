# This program is to compute power spectrum in different experiments
# Import package
import gc
import json
import numpy as np
import netCDF4 as nc

from scipy.signal import detrend
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt

exec(open("/home/b11209013/Package/WaveDispersion.py").read())
exec(open("/home/b11209013/Package/Plot_Style.py").read())
# function for computing power spectrum


def powerspectrum(data: np.ndarray) -> np.ndarray:

    data_fft = np.fft.fft(data, axis=1)
    data_fft = np.fft.ifft(data_fft, axis=-1) * data.shape[-1]

    ps = (data_fft * data_fft.conj()) / (data.shape[1]*data.shape[-1])**2.0

    return ps.mean(axis=0).real

# Compute vertically averaged power spectrum


def vert_int(
    data: np.ndarray,
    lev: np.ndarray,
) -> np.ndarray:
    return np.trapz(data, lev*100.0, axis=1) \
        / np.trapz(np.ones_like(lev), lev*100.0)

# compute background spectrum from symmetric and asymmetric data


def compute_background(data: np.ndarray) -> np.ndarray:
    data = data.copy()

    kernel = np.array([1, 2, 1]) / 4.0

    half_freq = data.shape[0]//2

    for i in range(10):
        data = convolve1d(data, kernel, axis=0, mode="reflect")

    for i in range(10):
        data[:half_freq] = convolve1d(
            data[:half_freq], kernel, axis=1, mode="reflect")

    for i in range(40):
        data[half_freq:] = convolve1d(
            data[half_freq:], kernel, axis=1, mode="reflect")

    return data


def remove_linear_trend(data: np.ndarray, axis: int) -> np.ndarray:
    # time or axis coordinate
    coords = np.arange(data.shape[axis])
    # bring trend axis to front
    data_swapped = np.moveaxis(data, axis, 0)
    # flatten all other dims
    flat = data_swapped.reshape(data.shape[axis], -1)

    # [slope, intercept], shape: (2, N)
    coefs = np.polyfit(coords, flat, 1)
    trend = np.outer(coords, coefs[0]) + \
        coefs[1]                # reconstruct trend

    detrended = flat - trend                                     # subtract trend
    # back to swapped shape
    detrended = detrended.reshape(data_swapped.shape)

    # restore original axis order
    return np.moveaxis(detrended, 0, axis)


def main():
    # ==== 1. Load data ==== #
    fpath: str = "/data92/b11209013/MPAS/merged_data/"

    exp_list: list[str] = ["CNTL", "NCRF"]

    # # Load dimension
    with nc.Dataset(f"{fpath}{exp_list[0]}/theta.nc") as ds:
        dims: dict[str, np.ndarray] = {
            key: ds[key][...]
            for key in ds.dimensions.keys()
            if not key == "time"
        }

    lat_lim: list[int] = np.where(
        (dims["lat"] >= -5.0) & (dims["lat"] <= 5.0))[0]

    dims["lat"] = dims["lat"][lat_lim]

    converter: np.ndarray = (
        1000.0 / dims["lev"][None, :, None, None]) ** (-0.285)
    # convert from theta to temperature

    # # Load data
    data: dict[str, np.ndarray] = {}

    for exp in exp_list:
        with nc.Dataset(f"{fpath}{exp}/theta.nc") as ds:

            data[exp] = ds["theta"][..., lat_lim, :] * converter

    # # load the last 360 days' data
    data = {
        exp: data[exp][-360:, ...]
        for exp in exp_list
    }

    print("Finished: loading data")

    # ==== 2. Processing data ==== #
    # # compute anomalies
    data_anomalies: dict[str, np.ndarray] = {
        exp: data[exp] - data[exp].mean(axis=(0, 3), keepdims=True)
        for exp in exp_list
    }

    del data
    gc.collect()

    print("Finished: compute anomalies")

    # # form symmetric data
    data_symm: dict[str, np.ndarray] = {
        exp: (data_anomalies[exp] + np.flip(data_anomalies[exp], axis=2)) / 2.0
        for exp in exp_list
    }

    data_asym: dict[str, np.ndarray] = {
        exp: (data_anomalies[exp] - np.flip(data_anomalies[exp], axis=2)) / 2.0
        for exp in exp_list
    }
    print("Finished: form data")

    # # chunking data into windows
    # # # setup hanning window
    hanning: np.ndarray = np.hanning(120)[:, None, None, None]

    symm_chunks: dict[str, np.ndarray] = {
        exp: np.array([
            detrend(data_symm[exp][i*60:i*60+120], 0) * hanning
            for i in range(360//120+2)
        ])
        for exp in exp_list
    }

    asym_chunks: dict[str, np.ndarray] = {
        exp: np.array([
            detrend(data_asym[exp][i*60:i*60+120], 0) * hanning
            for i in range(360//120+2)
        ])
        for exp in exp_list
    }

    del data_anomalies, data_symm, data_asym
    gc.collect()

    print("Finished: chunking data")

    # ==== 3. Compute Power Spectrum ==== #
    # # Compute power spectrum for each layer
    symm_power: dict[str, np.ndarray] = {
        exp: powerspectrum(symm_chunks[exp])
        for exp in exp_list
    }

    asym_power: dict[str, np.ndarray] = {
        exp: powerspectrum(asym_chunks[exp])
        for exp in exp_list
    }

    # # Vertically average power spectrum

    symm_vint: dict[str, np.ndarray] = {
        exp: np.fft.fftshift(
            vert_int(symm_power[exp], dims["lev"]).sum(axis=1))
        for exp in exp_list
    }

    asym_vint: dict[str, np.ndarray] = {
        exp: np.fft.fftshift(
            vert_int(asym_power[exp], dims["lev"]).sum(axis=1))
        for exp in exp_list
    }

    wn: np.ndarray = np.fft.fftshift(np.fft.fftfreq(
        dims["lon"].size, d=1/(dims["lon"].size))).astype(int)
    fr: np.ndarray = np.fft.fftshift(np.fft.fftfreq(120, d=1/4))

    symm_vint: dict[str, np.ndarray] = {
        exp: symm_vint[exp][fr > 0]
        for exp in exp_list
    }

    asym_vint: dict[str, np.ndarray] = {
        exp: asym_vint[exp][fr > 0]
        for exp in exp_list
    }

    # ==== 4. Compute background Spectrum and Peak ==== #
    # # Compute background
    background: dict[str, np.ndarray] = {
        exp: compute_background((symm_vint[exp] + asym_vint[exp]) / 2.0)
        for exp in exp_list
    }

    # ==== 5. Setup for plotting ==== #
    # # Compute dispersion relation
    dispersion = EquatorialWaveDispersion(
        nPlanetaryWave=100, Ahe=[100, 50, 25, 10])

    fr_ana, wn_ana = dispersion.compute()

    # # Specific dispersion relation function for KW
    def kel_curve(wn, ed): return 86400.0 * wn * \
        np.sqrt(9.81 * ed) / (2 * np.pi * 6.371e6)

    def inverse_kel(fr, ed): return fr * (2*np.pi*6.371e6) / \
        (86400 * np.sqrt(9.81 * ed))

    # # domain for Kelvin wave
    def kw_box(dis, inv_dis):
        plt.plot(
            np.linspace(1, 15),
            kel_curve(np.linspace(1, 15), 100),
            color="r", linestyle="-", linewidth=3
        )
        plt.plot(
            np.linspace(1, 15),
            kel_curve(np.linspace(1, 15), 10),
            color="r", linestyle="-", linewidth=3
        )

        plt.hlines(1/20, xmin=1, xmax=inv_dis(1/20, 10),
                   color="r", linestyle="-", linewidth=3)
        plt.hlines(1/2, xmin=inv_dis(1/2, 100), xmax=15,
                   color="r", linestyle="-", linewidth=3)
        plt.vlines(1, ymin=1/20, ymax=dis(1, 100),
                   color="r", linestyle="-", linewidth=3)
        plt.vlines(15, ymin=dis(15, 10), ymax=1/2,
                   color="r", linestyle="-", linewidth=3)
        plt.hlines(1/20, xmin=-15, xmax=15,
                   color="k", linestyle="--", linewidth=2)
        plt.hlines(1/8, xmin=-15, xmax=15,
                   color="k", linestyle="--", linewidth=2)
        plt.hlines(1/3, xmin=-15, xmax=15,
                   color="k", linestyle="--", linewidth=2)
        plt.text(15, 1/20, "20 days", ha="left", va="bottom")
        plt.text(15, 1/8, "8 days", ha="left", va="bottom")
        plt.text(15, 1/3, "3 days", ha="left", va="bottom")

    # # Start plotting
    apply_custom_plot_style()

    plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], symm_vint["CNTL"]/background["CNTL"],
                       cmap="Blues", vmin=1, vmax=5)
    for j in range(2, 6):
        for i in range(4):
            plt.plot(wn_ana[j, i], fr_ana[j, i], color="k",
                     linestyle="--", linewidth=2)
    kw_box(kel_curve, inverse_kel)
    plt.xticks(np.linspace(-14, 14, 8))
    plt.yticks(np.linspace(0, 0.5, 6))
    plt.xlim(-15, 15)
    plt.ylim(0, 0.5)
    plt.xlabel("Zoanl Wavenumber")
    plt.ylabel("Frequency [ CPD ]")
    plt.title("CNTL")
    plt.colorbar(c, label="Normalized Power Spectrum",
                 orientation="horizontal")
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/CNTL_power.png", dpi=500)
    plt.show()

    plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], symm_vint["NCRF"]/background["CNTL"],
                       cmap="Blues", vmin=1, vmax=5)
    for j in range(2, 6):
        for i in range(4):
            plt.plot(wn_ana[j, i], fr_ana[j, i], color="k",
                     linestyle="--", linewidth=2)
    kw_box(kel_curve, inverse_kel)
    plt.xticks(np.linspace(-14, 14, 8))
    plt.yticks(np.linspace(0, 0.5, 6))
    plt.xlim(-15, 15)
    plt.ylim(0, 0.5)
    plt.xlabel("Zoanl Wavenumber")
    plt.ylabel("Frequency [ CPD ]")
    plt.title("NCRF")
    plt.colorbar(c, label="Normalized Power Spectrum",
                 orientation="horizontal")

    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/NCRF_power.png", dpi=500)
    plt.show()

    # compute sum of kelvin wave variance;
    wnm, frm = np.meshgrid(wn, fr[fr > 0])

    kel_cond = np.where(
        (wnm >= 1) & (wnm <= 15) &
        (frm >= 1/20) & (frm <= 1/2) &
        (frm >= kel_curve(wnm, 10)) & (frm <= kel_curve(wnm, 100)), 1.0, np.nan
    )

    variance_sum = {
        exp: np.nanmean(symm_vint[exp] * kel_cond)
        for exp in exp_list
    }

    with open("/home/b11209013/2025_Research/AOGS/File/variance_sum.json", "w") as f:
        json.dump(variance_sum, f, indent=4)


if __name__ == '__main__':
    main()
