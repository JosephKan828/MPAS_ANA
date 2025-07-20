# This program is to apply EOF analysis on CNTL and NCRF data

import numpy as np
import netCDF4 as nc
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

exec(open("/home/b11209013/Package/Plot_Style.py").read())

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend explicitly

def main():
    
    # ==== 1. load data ==== #
    fpath: str = "/data92/b11209013/MPAS/merged_data"
    
    exp_list: list[str] = ["CNTL", "NCRF"]
    
    data = {}
    # load dimensions
    with nc.Dataset(fpath + "/CNTL/q1.nc") as ds:
        dims = {key: ds[key][:] for key in ds.dimensions.keys()}
        
        lat_lim: np.ndarray = np.where((dims["lat"] >= -5.0) & (dims["lat"] <= 5.0))[0]
        dims["lat"] = dims["lat"][lat_lim]
        dims["time"] = dims["time"][-360:]
        
        data["CNTL"] = ds["q1"][-360:, :, lat_lim, :].transpose(1, 0 ,2, 3).reshape(len(dims["lev"]), -1)
    
    with nc.Dataset(fpath + "/NCRF/q1.nc") as ds:
        data["NCRF"] = ds["q1"][-360:, :, lat_lim, :].transpose(1, 0 ,2, 3).reshape(len(dims["lev"]), -1)
    

    # ==== 2. apply EOF analysis ==== #
    # compute anomalies
    data_anom = {
        exp: data[exp] - np.mean(data[exp], axis=1, keepdims=True)
        for exp in exp_list
    }
    
    # apply PCA
    data_pca = {}
    eof = {}
    exp_var = {}
    
    for exp in exp_list:
        obj = PCA(n_components=2)
        
        data_pca[exp] = obj.fit(data_anom[exp].T)
        eof[exp] = obj.components_.T
        exp_var[exp] = obj.explained_variance_ratio_
    
    print(exp_var)

    apply_custom_plot_style() # type: ignore
    
    
    fig = plt.figure(figsize=(11, 16))
    axes = plt.gca()
    plt.plot(eof["CNTL"][:, 0], dims["lev"], label=f"CNTL: {exp_var['CNTL'][0]:.2f}")
    plt.plot(eof["NCRF"][:, 0], dims["lev"], label=f"NCRF: {exp_var['NCRF'][0]:.2f}")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    plt.axvline(0, color="k", linewidth=0.5, linestyle="--")
    plt.yscale("log")
    plt.legend(fontsize=32)
    plt.yticks([200, 300, 400, 500, 600, 800, 1000],
                ["200", "300", "400", "500", "600", "800", "1000"]
                )
    plt.xlim(-0.32, 0.32)
    plt.ylim(1000, 150)
    plt.ylabel("Pressure (hPa)")
    plt.title("EOF1 of q1")
    plt.grid(False)
    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EOF1.png")
    plt.close(fig)
    
    fig = plt.figure(figsize=(11, 16))
    axes = plt.gca()
    plt.plot(eof["CNTL"][:, 1], dims["lev"], label=f"CNTL: {exp_var['CNTL'][1]:.2f}")
    plt.plot(eof["NCRF"][:, 1], dims["lev"], label=f"NCRF: {exp_var['NCRF'][1]:.2f}")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    plt.axvline(0, color="k", linewidth=0.5, linestyle="--")
    plt.yscale("log")
    plt.legend(fontsize=32)
    plt.yticks([200, 300, 400, 500, 600, 800, 1000],
                ["200", "300", "400", "500", "600", "800", "1000"]
                )
    plt.xlim(-0.32, 0.32)
    plt.ylim(1000, 150)
    plt.ylabel("Pressure (hPa)")
    plt.title("EOF2 of q1")
    plt.grid(False)
    plt.savefig("/home/b11209013/2025_Research/AOGS/Figure/EOF2.png")
    plt.close(fig)
    
if __name__ == "__main__":
    main()