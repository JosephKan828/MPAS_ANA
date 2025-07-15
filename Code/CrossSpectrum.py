# This program is to compute power spectrum in different experiments
# Import package
import gc
import json
import numpy as np
import netCDF4 as nc

from itertools import product
from scipy.signal import detrend
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

exec(open("/home/b11209013/Package/WaveDispersion.py").read())
exec(open("/home/b11209013/Package/Plot_Style.py").read())
# function for computing power spectrum


def crossspectrum(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:

    data_fft1 = np.fft.fft(data1, axis=1)
    data_fft1 = np.fft.ifft(data_fft1, axis=-1) * data1.shape[-1]

    data_fft2 = np.fft.fft(data2, axis=1)
    data_fft2 = np.fft.ifft(data_fft2, axis=-1) * data2.shape[-1]

    ps = (data_fft1 * data_fft2.conj()) / (data1.shape[1]*data2.shape[-1])**2.0

    ps_conv = convolve1d(ps, np.array([1, 2, 1])/4, axis=1, mode="reflect")
    ps_conv = convolve1d(ps_conv, np.array(
        [1, 2, 1])/4, axis=-1, mode="reflect")

    return ps_conv.mean(axis=0).real

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


def compute_stability(data: np.ndarray, lev: np.ndarray) -> np.ndarray:

    mean_temp = data.mean(axis=(0, 3))

    temp_grad = np.gradient(mean_temp, lev*100.0, axis=0)

    lev_bc = lev[:, None]*100.0

    stab = 9.81/287.5 * (lev_bc/mean_temp) * temp_grad

    return 9.81/1004.5 - stab


def main():
    # ==== 1. Load data ==== #
    fpath: str = "/data92/b11209013/MPAS/merged_data/"

    exp_list: list[str] = ["CNTL", "NCRF"]
    var_list: list[str] = ["q1", "w", "theta"]

    iter_list: list[str] = list(product(var_list, exp_list))

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
    data: dict[str, np.ndarray] = {
        var: {} for var in var_list
    }

    for iter in iter_list:
        var, exp = iter

        with nc.Dataset(f"{fpath}{exp}/{var}.nc") as ds:

            if var == "theta":
                data[var][exp] = ds[var][..., lat_lim, :][-360:, ...] * converter

            else:
                data[var][exp] = ds[var][..., lat_lim, :][-360:, ...]

    print("Finished: loading data")

    # ==== 2. Processing data ==== #
    # # Compute stability
    stability: dict[str, np.ndarray] = {
        exp: compute_stability(data["theta"][exp], dims["lev"])
        for exp in exp_list
    }

    # # compute anomalies
    data_anomalies: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: data[var][exp] -
            data[var][exp].mean(axis=(0, 3), keepdims=True)
            for exp in exp_list
        }
        for var in var_list
    }

    del data
    gc.collect()

    data_anomalies["temp_tend"] = {
        exp: np.gradient(data_anomalies["theta"][exp], 6*3600.0, axis=0)
        for exp in exp_list
    }

    var_list = data_anomalies.keys()

    print("Finished: compute anomalies")

    # # form symmetric data
    data_symm: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: (data_anomalies[var][exp] +
                  np.flip(data_anomalies[var][exp], axis=2)) / 2.0
            for exp in exp_list
        }
        for var in var_list
    }

    print("Finished: form data")

    # # chunking data into windows
    # # # setup hanning window
    hanning: np.ndarray = np.hanning(120)[:, None, None, None]

    symm_chunks: dict[str, dict[str, np.ndarray]] = {
        var: {
            exp: np.array([
                detrend(data_symm[var][exp][i*60:i*60+120], 0) * hanning
                for i in range(360//120+2)
            ])
            for exp in exp_list
        }
        for var in var_list
    }

    del data_anomalies, data_symm
    gc.collect()

    print("Finished: chunking data")

    # ==== 3. Compute Power Spectrum ==== #
    # # Compute power spectrum for each layer
    qt_cross: dict[str, np.ndarray] = {
        exp: crossspectrum(symm_chunks["q1"][exp],
                           symm_chunks["theta"][exp]) / 1004.5
        for exp in exp_list
    }

    wt_cross: dict[str, np.ndarray] = {
        exp: crossspectrum(stability[exp][None, :, :, None] * symm_chunks["w"][exp],
                           symm_chunks["theta"][exp])
        for exp in exp_list
    }

    tdt_cross: dict[str, np.ndarray] = {
        exp: crossspectrum(symm_chunks["temp_tend"][exp],
                           symm_chunks["theta"][exp])
        for exp in exp_list
    }

    # # Vertically average power spectrum
    wn: np.ndarray = np.fft.fftshift(np.fft.fftfreq(
        dims["lon"].size, d=1/(dims["lon"].size))).astype(int)
    fr: np.ndarray = np.fft.fftshift(np.fft.fftfreq(120, d=1/4))

    qt_vint: dict[str, np.ndarray] = {
        exp: np.fft.fftshift(
            vert_int(qt_cross[exp], dims["lev"])).sum(axis=1)[fr > 0] * 86400.0
        for exp in exp_list
    }

    wt_vint: dict[str, np.ndarray] = {
        exp: np.fft.fftshift(
            vert_int(wt_cross[exp], dims["lev"])).sum(axis=1)[fr > 0] * 86400.0
        for exp in exp_list
    }

    tdt_vint: dict[str, np.ndarray] = {
        exp: np.fft.fftshift(
            vert_int(tdt_cross[exp], dims["lev"])).sum(axis=1)[fr > 0] * 86400.0
        for exp in exp_list
    }

    # ==== 4. Setup for plotting ==== #
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
            np.linspace(inv_dis(1/20, 10), 15),
            kel_curve(np.linspace(inv_dis(1/20, 10), 15), 10),
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
        plt.text(15, 1/20, "20 days", ha="right", va="bottom")
        plt.text(15, 1/8, "8 days", ha="right", va="bottom")
        plt.text(15, 1/3, "3 days", ha="right", va="bottom")

    # # Start plotting
    apply_custom_plot_style()

    fig = plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], qt_vint["CNTL"],
                       cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0))
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
    cbar = plt.colorbar(c, label=r"$\frac{1}{C_p} Q_1^\prime T^\prime$ [ $K^2/day$ ]",
                        orientation="horizontal")
    cbar.set_ticks(np.linspace(-0.015, 0.005, 5))
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/CNTL_qt.png", dpi=500)
    # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], qt_vint["NCRF"],
                       cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0))
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
    cbar = plt.colorbar(c, label=r"$\frac{1}{C_p} Q_1^\prime T^\prime$ [ $K^2/day$ ]",
                        orientation="horizontal")
    cbar.set_ticks([-0.020, -0.010, 0, 0.005, 0.010])
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/NCRF_qt.png", dpi=500)
    # plt.show(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], wt_vint["CNTL"],
                       cmap="BrBG_r",
                       norm=TwoSlopeNorm(vcenter=0))
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
    cbar = plt.colorbar(c,
                        label=r"$\left(\Gamma_d - \overline{\Gamma}\right) w^\prime T^\prime$ [ $K^2/day$ ]",
                        orientation="horizontal")
    cbar.set_ticks(np.linspace(-0.020, 0.005, 6))
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/CNTL_wt.png", dpi=500)
    # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], wt_vint["NCRF"],
                       cmap="BrBG_r",
                       norm=TwoSlopeNorm(vcenter=0))
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
    cbar = plt.colorbar(c,
                        label=r"$\left(\Gamma_d - \overline{\Gamma}\right) w^\prime T^\prime$ [ $K^2/day$ ]",
                        orientation="horizontal")
    cbar.set_ticks([-0.03, -0.02, -0.01, 0, 0.005, 0.01])
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/NCRF_wt.png", dpi=500)
    # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], tdt_vint["CNTL"],
                       cmap="PiYG_r",
                       norm=TwoSlopeNorm(vcenter=0))
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
    cbar = plt.colorbar(c,
                        label=r"$T^\prime \frac{\partial T^\prime}{\partial t}$ [ $K^2/day$ ]",
                        orientation="horizontal")
    cbar.set_ticks(np.linspace(-0.0003, 0.0003, 7))
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/CNTL_tdt.png", dpi=500)
    # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(18, 16.5))
    c = plt.pcolormesh(wn, fr[fr > 0], tdt_vint["NCRF"],
                       cmap="PiYG_r",
                       norm=TwoSlopeNorm(vcenter=0))
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
    cbar = plt.colorbar(c,
                        label=r"$T^\prime \frac{\partial T^\prime}{\partial t}$ [ $K^2/day$ ]",
                        orientation="horizontal")
    cbar.set_ticks([-0.0004, -0.0002, 0, 0.0001])
    plt.savefig(
        "/home/b11209013/2025_Research/AOGS/Figure/NCRF_tdt.png", dpi=500)
    # plt.show()
    plt.close(fig)

    # compute sum of kelvin wave variance;
    wnm, frm = np.meshgrid(wn, fr[fr > 0])

    kel_cond = np.where(
        (wnm >= 1) & (wnm <= 15) &
        (frm >= 1/20) & (frm <= 1/2) &
        (frm >= kel_curve(wnm, 10)) & (frm <= kel_curve(wnm, 100)), 1.0, np.nan
    )

    qt_mean = {
        exp: np.nanmean(qt_vint[exp] * kel_cond)
        for exp in exp_list
    }

    wt_mean = {
        exp: np.nanmean(wt_vint[exp] * kel_cond)
        for exp in exp_list
    }

    tdt_mean = {
        exp: np.nanmean(tdt_vint[exp] * kel_cond)
        for exp in exp_list
    }

    with open("/home/b11209013/2025_Research/AOGS/File/qt_mean.json", "w") as f:
        json.dump(qt_mean, f, indent=4)

    with open("/home/b11209013/2025_Research/AOGS/File/wt_mean.json", "w") as f:
        json.dump(wt_mean, f, indent=4)

    with open("/home/b11209013/2025_Research/AOGS/File/tdt_mean.json", "w") as f:
        json.dump(tdt_mean, f, indent=4)


if __name__ == '__main__':
    main()
