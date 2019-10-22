# -*- coding: utf-8 -*-
"""Visualization functions."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg") # use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib import rc
import seaborn as sns


ALPHA = 0.35

# set style
sns.set_style("ticks")
rc("font", size=12.5)
rc("text", usetex=True)
rc("grid", linestyle="dotted")
rc("axes", unicode_minus=False)
rc("axes", labelsize=15)  
rc("axes", titlesize=15)  
rc("axes", linewidth=1.2)
rc("legend", fontsize=12.5)
rc("legend", handlelength=1)
rc("xtick", labelsize=12.5)
rc("ytick", labelsize=12.5)
rc("figure", figsize=(5, 4))
rc("lines", linewidth=1.2)


def plot_weight_distr(names, qm_vals, qs_vals):
    """Plot weight distribution.
        
    Parameters
    ----------
    names : [str]
        List of layer names.
    qm_vals : [ndarray]
        List of 1D array of `float` representing layer weight means.
    qs_vals : [ndarray]
        List of 1D array of `float` representing layer weight standard deviations.
    """
    assert len(qm_vals) == len(qs_vals)
    
    fig = figure.Figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.kdeplot(qm.flatten(), ax=ax, label=n, shade=True)
    ax.grid(True, linewidth=1.5)
    ax.set_xlabel("Weights mean")
    ax.set_ylabel("Density")
    ax.set_xlim([-1.5, 1.5])
    ax.get_legend().remove()
    sns.despine(ax=ax)

    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.kdeplot(qs.flatten(), ax=ax, label=n.replace("Linear", "Dense"), shade=True)
        if sum(qs.flatten()) == 0:
            ax.axvline(x=0, linestyle="-")

    ax.grid(True, linewidth=1.5)
    ax.set_xlabel("Weights standard deviation")
    ax.set_ylabel("Density")
    ax.set_xlim([0, 1.])
    ax.legend(loc="upper right", frameon=True)
    sns.despine(ax=ax)

    fig.tight_layout()

    return fig


def plot_predictive_distr(preds, mean, std):
    """Plot predictive distribution.
        
    Parameters
    ----------
    preds : ndarray
        1D array of `float` representing the predicted labels.
    mean : float
        Predictive mean.
    std : float
        Predictive standard deviation.
    """
    RED = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)

    fig = plt.figure()

    p = sns.kdeplot(preds, shade=True)
    x, y = p.get_lines()[0].get_data()

    plt.axvline(x=mean, linestyle="--", color=RED, linewidth=1, label=r"$ \mu \enspace = %.2f$" % mean)
    plt.fill_between(x, 0, y, where=((x > mean - std) & (x < mean + std)), color=RED, alpha=ALPHA)
    plt.fill(np.NaN, np.NaN, color=RED, alpha=ALPHA, label=r"$2\sigma = %.2f$" % (2 * std))

    plt.grid(True, linewidth=1.5)
    plt.xlabel("Predicted RUL")
    plt.ylabel("Density")
    plt.legend(loc="upper left", frameon=True)
    sns.despine()
    
    fig.tight_layout()

    return fig
