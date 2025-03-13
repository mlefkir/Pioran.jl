
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import scipy.stats as stats

plt.style.use(
    "https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs_colblind.mplstyle"
)

def replot_julia_results(t,y,yerr,path):

    f_min = 1/(t[-1]-t[0])
    f_max = 1/(2*np.min(np.diff(t)))

    if os.path.isfile("{path}psd_ppc_data.txt"):
        A = np.genfromtxt(f"{path}psd_ppc_data.txt").T
        psd_noise_levels = np.genfromtxt(f"{path}psd_noise_levels.txt")
        f = A[:, 0]
        psd_quantiles = A[:, 1:6]
        psd_approx_quantiles = A[:, 6:]
        replot_psd_ppc(f, psd_quantiles, psd_approx_quantiles, psd_noise_levels, f_min, f_max,path)

    if os.path.isfile(f"{path}ppc_timeseries_quantiles.txt"):
        ts_quantiles = np.genfromtxt(f"{path}ppc_timeseries_quantiles.txt").T
        t_pred = np.genfromtxt(f"{path}ppc_t_pred.txt")
        replot_ts_ppc(t, y, yerr, t_pred, ts_quantiles, path)

    if os.path.isfile(f"{path}ppc_residuals_quantiles.txt"):
        res_quantiles = np.genfromtxt(f"{path}ppc_residuals_quantiles.txt").T
        ppc_residuals_mean = np.genfromtxt(f"{path}ppc_residuals_mean.txt")
        A = np.genfromtxt(f"{path}ppc_residuals_acvf.txt").T
        lags = A[:,0]
        acvf_mean = A[:,2]
        replot_residuals_ppc(t, ppc_residuals_mean, res_quantiles, lags, acvf_mean,path)

    if os.path.isfile(f"{path}binned_lsp_data.txt"):
        lsp_data = np.genfromtxt(f"{path}binned_lsp_data.txt")
        lsp_ppc_data = np.genfromtxt(f"{path}lsp_ppc_data.txt").T
        replot_lsp_ppc(lsp_data,lsp_ppc_data,f_min,f_max,path)

def replot_lsp_ppc(lsp_data,lsp_ppc_data,f_min,f_max,path):
    """Replot the Lomb-Scargle periodogram PPC.

    Parameters
    ----------
    lsp_data : array
        The binned Lomb-Scargle periodogram data. [f, lsp]
    lsp_ppc_data : array
        The Lomb-Scargle periodogram PPC data. [f, lsp_quantiles]
    f_min : float
        The minimum frequency.
    f_max : float
        The maximum frequency.
    path : str
        The path to save the plot.

    """
    binned_f = lsp_data[:,0]
    binned_lsp = lsp_data[:,1]
    f = lsp_ppc_data[:,0]
    lsp_quantiles = lsp_ppc_data[:,1:]

    fig,ax = plt.subplots(1,1,figsize=(6,3.5))


    ax.loglog(f,lsp_quantiles[:,2],label="Median realisations",color="C0",alpha=0.5)
    ax.fill_between(f,lsp_quantiles[:,1],lsp_quantiles[:,3],color="C2",alpha=0.3,label="68%")
    ax.fill_between(f,lsp_quantiles[:,0],lsp_quantiles[:,4],color="C2",alpha=0.15,label="95%")
    ax.axvline(f_min,color="k",ls=":",label=r"$f_{\min}, f_{\max}$")
    ax.axvline(f_max,color="k",ls=":")
    ax.loglog(binned_f,binned_lsp,label="Data",color="C4")
    ax.set_xlim(np.min(f),np.max(f))
    ax.update({'xlabel': r'Frequency $(d^{-1})$', 'ylabel': 'Lomb-Scargle power'})


    ax.legend(
        ncol=2,
        bbox_to_anchor=(0.5, -0.4),
        loc="lower center",
        bbox_transform=fig.transFigure,
    )
    fig.savefig(f"{path}replot_lsp_ppc.pdf",bbox_inches="tight")

def replot_psd_ppc(f, psd_quantiles, psd_approx_quantiles, psd_noise_levels, f_min, f_max, path
):
    """Replot the PSD PPC plot.

    Parameters
    ----------
    f : array
        The frequency array.
    psd_quantiles : array
        The quantiles of the PSD model.
    psd_approx_quantiles : array
        The quantiles of the PSD approximation.
    psd_noise_levels : array
        The noise levels.
    f_min : float
        The minimum frequency.
    f_max : float
        The maximum frequency.
    path : str
        The path to save the figure.
    """
    approx_color = "C6"
    psd_color = "C3"
    noise_color = "C5"
    window_color = "k"

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.loglog(f, psd_quantiles[:, 2], label="Median", color=psd_color)
    ax.fill_between(
        f, psd_quantiles[:, 1], psd_quantiles[:, 3], color=psd_color, alpha=0.3
    )
    ax.fill_between(
        f, psd_quantiles[:, 0], psd_quantiles[:, 4], color=psd_color, alpha=0.15
    )
    ax.axhline(psd_noise_levels[0], color=noise_color, ls="-", label="Noise level")
    ax.loglog(f, psd_approx_quantiles[:, 2], color=approx_color)
    ax.fill_between(
        f,
        psd_approx_quantiles[:, 1],
        psd_approx_quantiles[:, 3],
        color=approx_color,
        alpha=0.3,
    )
    ax.fill_between(
        f,
        psd_approx_quantiles[:, 0],
        psd_approx_quantiles[:, 4],
        color=approx_color,
        alpha=0.15,
    )
    ax.axvline(f_min, color=window_color, ls=":", label=r"f$_{\min}$")
    ax.axvline(f_max, color=window_color, ls=":", label=r"f$_{\max}$")
    ax.update({"xlabel": r"Frequency $(d^{-1})$", "ylabel": "Power Spectral Density"})
    ax.set_xlim(np.min(f), np.max(f) / 10)
    ax.set_ylim(np.min(psd_noise_levels) / 10)

    legend_elements = [
        Line2D([0], [0], color=psd_color, lw=2, label="PSD model"),
        Line2D([0], [0], color=approx_color, lw=2, label="PSD approximation"),
        Line2D([0], [0], color=noise_color, lw=2, label="Noise level"),
        Patch(facecolor="k", edgecolor="k", alpha=0.1, label="95%"),
        Patch(facecolor="k", edgecolor="k", alpha=0.4, label="68%"),
        Line2D(
            [0], [0], color=window_color, lw=2, ls=":", label="$f_{\min}, f_{\max}$"
        ),
    ]

    ax.legend(
        handles=legend_elements,
        ncol=2,
        bbox_to_anchor=(0.5, -0.175),
        loc="lower center",
        bbox_transform=fig.transFigure,
    )
    fig.tight_layout()
    fig.savefig(f"{path}replot_psd_ppc.pdf", bbox_inches="tight")


def replot_ts_ppc(t, y, yerr, t_pred, ts_quantiles, path):
    """Replot the time series ppc with the data and the quantiles

    Parameters
    ----------
    t : array
        Time of the data
    y : array
        Flux of the data
    yerr : array
        Error of the data
    t_pred : array
        Time of the prediction
    ts_quantiles : array
        Quantiles of the time series prediction
    path : str
        Path to save the figure
    """
    data_color = "k"
    fill_color = "C6"

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

    ax.margins(x=0.025, y=0.025)
    ax.plot(t_pred, ts_quantiles[:, 2], label="Median", color="C6")
    ax.errorbar(
        t,
        y,
        yerr,
        fmt="o",
        label="Data",
        color=data_color,
        ms=3,
        mfc="grey",
        elinewidth=1,
        zorder=10,
        mew=0.65,
    )
    ax.fill_between(
        t_pred,
        ts_quantiles[:, 1],
        ts_quantiles[:, 3],
        color=fill_color,
        alpha=0.4,
        label=r"$68\%$",
    )
    ax.fill_between(
        t_pred,
        ts_quantiles[:, 0],
        ts_quantiles[:, 4],
        color=fill_color,
        alpha=0.2,
        label=r"$95\%$",
    )
    ax.update({"xlabel": r"Time ($d$)", "ylabel": "Time series"})
    ax.legend(
        ncol=2,
        bbox_to_anchor=(0.5, -0.095),
        loc="lower center",
        bbox_transform=fig.transFigure,
    )

    fig.tight_layout()
    fig.savefig(f"{path}replot_ts_ppc.pdf", bbox_inches="tight")


def replot_residuals_ppc(t, ppc_residuals_mean, res_quantiles, lags, acvf_mean,path):
    """Replot the residuals and the autocorrelation of the residuals.

    Parameters
    ----------
    t : array
        Time array.
    ppc_residuals_mean : array
        Mean of the residuals.
    res_quantiles : array
        Quantiles of the residuals.
    lags : array
        Lags for the autocorrelation.
    acvf_mean : array
        Mean of the autocorrelation.
    path : str

    """

    n = len(t)

    res_color = "C0"
    res_sec_color =  "C5"

    confidence_intervals = [95,99]
    sigs = [stats.norm.ppf((50 + ci / 2) / 100) for ci in confidence_intervals]

    fig = plt.figure(tight_layout=True, figsize=(6, 4))

    gs0 = fig.add_gridspec(2, 1,hspace=0.45)
    gs00 = gs0[0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0)
    gs01 = gs0[1].subgridspec(1, 1)
    ax = []
    ax.append([fig.add_subplot(gs00[0, 0]), fig.add_subplot(gs00[0, 1])])
    ax.append(fig.add_subplot(gs01[0, :]))
    ax[0][0].sharey(ax[0][1])
    ax[0][0].scatter(t, ppc_residuals_mean, facecolor="w",s=9, linewidths=.75,label="Residuals",ec=res_sec_color, zorder=10)
    ax[0][0].axhline(0, c="k", ls="-",zorder=10,lw=1)
    ax[0][0].set_ylabel("Residuals")
    ax[0][0].fill_between(t, res_quantiles[:, 1], res_quantiles[:, 3], alpha=0.3,color=res_color)
    ax[0][0].fill_between(t, res_quantiles[:, 0], res_quantiles[:, 4], alpha=0.3,color=res_color)
    ax[0][0].set_xlabel("Time ($d$)")

    ax[0][1].hist(ppc_residuals_mean, bins=20,alpha=0.5, orientation="horizontal", density=True, color=res_sec_color)
    ax[0][1].axhline(0, c="k", ls="-",lw=1)
    ax[0][1].set_xticks([])
    ax[0][1].tick_params(axis="y", labelleft=False,top=False, bottom=False, left=False, right=False)
    ax[0][1].spines["right"].set_visible(False)
    ax[0][1].spines["top"].set_visible(False)
    ax[0][1].spines["bottom"].set_visible(False)
    x = np.linspace(-4, 4, 100)
    pdf = stats.norm.pdf(x,*stats.norm.fit(ppc_residuals_mean))
    ax[0][1].plot(pdf,x, color=res_color)


    markerline, stemline, baseline, = ax[1].stem(lags, acvf_mean, label="Mean", basefmt=" ", linefmt="-", markerfmt="C3")
    plt.setp(stemline, linewidth = 1.25, color = "C3")
    plt.setp(markerline, markersize = 4,color="C5",mec="k")


    for i,sig in enumerate(sigs):
        ax[1].fill_between(lags, -sig/np.sqrt(n), sig /np.sqrt(n),color="C3", alpha=0.2, label=f"{confidence_intervals[i]}% CI")
    ax[1].axhline(0, c="k", ls="-",lw=1)
    ax[1].set_ylabel("Autocorrelation")
    ax[1].set_xlabel("Lag (indices)")

    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{path}replot_residuals_ppc.pdf",bbox_inches="tight")

