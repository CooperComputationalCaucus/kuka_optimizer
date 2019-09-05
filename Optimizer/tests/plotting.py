from Optimizer.bayes_opt import UtilityFunction

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib
from matplotlib.font_manager import FontProperties

matplotlib.matplotlib_fname()
matplotlib.font_manager._rebuild()
font = {'size': 18}
matplotlib.rc('font', **font)

def check_dist(gp, a, b):
    threshold = 0.6
    if gp.kernel_(a.reshape(1, -1), b.reshape(1, -1)) > threshold:
        return True
    else:
        return False


def posterior(gp, grid):
    mu, sigma = gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp_1d(gp, df, axis, vector=None, utility_function=None, path=None, dpi=300):
    """

    Parameters
    ----------
    gp: gaussian process
    df: dataframe of X data with labels, and y data labeled Target
    axis: axis of reference by string or index
    vector: If given, must correspond to indexing of dataframe
    utility_function: instance of UtilityFunction, default to ucb ('greedy') 2.5.
    path: path for plot saving if desired
    dpi: dots per inch for output figure

    Returns
    -------
    fig

    Outputs
    -------
    Figure as png if given path
    """
    # Proccess dataframe
    X = df.drop(columns=['Target'])
    features = list(X.columns)
    y = df.Target
    if type(axis) == str:
        axis = features.index(axis)
    else:
        axis = int(axis)
    x_max = np.max(X.iloc[:, axis])
    x_min = np.min(X.iloc[:, axis])
    y_min = np.min(y)
    y_max = np.max(y)

    if not vector:
        vector = np.zeros(X.shape[1])
    else:
        vector = np.array(vector)

    if not utility_function:
        utility_function = UtilityFunction(kind="ucb", kappa=2.5, xi=0)

    # Prep data for plotting
    mask = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        vector[axis] = X.iloc[i, axis]
        mask[i] = check_dist(gp, np.array(X.iloc[i]), np.array(vector))
    x_near = X[mask]
    y_near = y[mask]
    x_far = X[(mask != True)]
    y_far = y[(mask != True)]
    x_grid = np.repeat(vector.reshape(1, -1), 10000, axis=0)
    x_linspace = np.linspace(x_min, x_max, 10000)
    x_grid[:, axis] = x_linspace
    x_linspace.reshape(-1, 1)
    mu, sigma = posterior(gp, x_grid)

    # Prep Figure
    fig = plt.figure(figsize=(16, 10))
    plt.rcParams['font.sans-serif'] = "DejaVu Sans"
    plt.rcParams['font.family'] = "sans-serif"
    fp = FontProperties(family="sans-serif", size=24, weight="bold")
    vector[axis] = -1
    fig.suptitle('Plot along {}'.format(vector)).set_fontproperties(fp)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    # Plot prediction
    ax.plot(x_linspace, mu, '--', linewidth=3, color='k', label='Prediction')
    ax.fill(np.concatenate([x_linspace, x_linspace[::-1]]),
            np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
            alpha=.6, fc='c', ec='None', label='95% confidence interval')
    # Plot observations
    ax.plot(x_near.iloc[:, axis], y_near, 'D', markersize=8, label='Local Observations', color='r')
    ax.plot(x_far.iloc[:, axis], y_far, 'o', markersize=1, label='Distant Observations', color='b')

    # Plot settings
    vector[axis] = -1
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max * 1.1))
    ax.set_ylabel('Hydrogen generation', fontdict={'size': 20})
    ax.set_xlabel('{}'.format(features[axis]), fontdict={'size': 16})

    # Plot Utility
    utility = utility_function.utility(x_grid, gp, 0)
    acq.plot(x_linspace, utility, label='Utility Function', color='green')
    acq.plot(x_linspace[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((x_min, x_max))
    acq.set_ylim((np.min(utility) * 0.9, np.max(utility) * 1.1))
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('{}'.format(features[axis]), fontdict={'size': 16})

    ax.legend(loc=1, borderaxespad=0.5)
    acq.legend(loc=1, borderaxespad=0.5)
    if path: plt.savefig(path, dpi=300)

    return fig


def plot_gp_2d(gp, df, a1, a2, vector=None, utility_function=None, path=None, dpi=300):
    """
    Plots gp along a specific from complete dataset along a specified
    vector in space for a given axis. If no vector is given, a zero vector
    is assumed.
    Points near the vector with respect to the kernel similarity are given
    ploted in scatter.

    Parameters
    ==========
    gp: gaussian process
    df: dataframe of X data with labels, and y data labeled Target
    ax1: axis of reference by string or index
    ax2: second axis of reference by string or index
    vector: If given, must correspond to indexing of dataframe
    utility_function: instance of UtilityFunction, default to ucb ('greedy') 2.5.
    path: path for plot saving if desired
    dpi: dots per inch for figure output

    Returns
    -------
    fig

    Outputs
    -------
    Figure as png if given path
    """
    n_points = 100
    # Proccess dataframe
    X = df.drop(columns=['Target'])
    features = list(X.columns)
    y = df.Target
    if type(a1) == str:
        a1 = features.index(a1)
    else:
        a1 = int(a1)
    if type(a2) == str:
        a2 = features.index(a2)
    else:
        a2 = int(a2)
    x1_max = np.max(X.iloc[:, a1])
    x2_max = np.max(X.iloc[:, a2])
    x1_min = np.min(X.iloc[:, a1])
    x2_min = np.min(X.iloc[:, a2])
    y_min = np.min(y)
    y_max = np.max(y)

    if not vector:
        vector = np.zeros(X.shape[1])
    else:
        vector = np.array(vector)

    if not utility_function:
        utility_function = UtilityFunction(kind="ucb", kappa=2.5, xi=0)

    # Prep data for plotting
    mask = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        vector[a1] = X.iloc[i, a1]
        vector[a2] = X.iloc[i, a2]
        mask[i] = check_dist(gp, np.array(X.iloc[i]), np.array(vector))
    x_near = X[mask]
    y_near = y[mask]
    x_far = X[(mask != True)]
    y_far = y[(mask != True)]

    # Creating complex mesh, unrolling, then predicting
    x1_mesh, x2_mesh = np.meshgrid(np.linspace(x1_min, x1_max, n_points), np.linspace(x2_min, x2_max, n_points))
    xx_roll = np.stack([np.ravel(x1_mesh), np.ravel(x2_mesh)], axis=1)
    x_grid = np.repeat(vector.reshape(1, -1), xx_roll.shape[0], axis=0)
    x_grid[:, a1] = xx_roll[:, 0]
    x_grid[:, a2] = xx_roll[:, 1]

    mu, sigma = posterior(gp, x_grid)
    mu_mesh = mu.reshape(x1_mesh.shape)
    sigma_mesh = sigma.reshape(x1_mesh.shape)
    utility = utility_function.utility(x_grid, gp, 0)
    u_mesh = utility.reshape(x1_mesh.shape)

    # Prep Figure
    fig = plt.figure(figsize=(20, 20))
    plt.rcParams['font.sans-serif'] = "DejaVu Sans"
    plt.rcParams['font.family'] = "sans-serif"
    ax = plt.subplot(projection='3d')

    # Plot prediction
    sig_col = cm.PuRd(sigma_mesh)
    # Right main plot
    smr = cm.ScalarMappable(cmap=cm.PuRd)
    smr.set_array(sigma_mesh)
    surf = ax.plot_surface(x1_mesh, x2_mesh, mu_mesh, facecolors=sig_col,
                           linewidth=0, antialiased=False, rstride=4, cstride=8, alpha=0.5)
    # Bottom
    smb = cm.ScalarMappable(cmap=cm.coolwarm)
    smb.set_array(u_mesh)
    contb = ax.contourf(x1_mesh, x2_mesh, u_mesh, zdir='z',
                        offset=0, levels=[i for i in range(int(y_max * 1.5))], cmap=cm.coolwarm)
    # Top (bar on left)
    smt = cm.ScalarMappable(cmap=cm.Spectral)
    smt.set_array(mu_mesh)
    contt = ax.contour(x1_mesh, x2_mesh, mu_mesh, zdir='z',
                       offset=y_max * 1.1, levels=[i for i in range(int(y_max * 1.5))], cmap=cm.Spectral)

    ax.scatter(x_near.iloc[:, a1], x_near.iloc[:, a2], marker='.', c='k')
    # ax.scatter(x_near.iloc[:,a1],x_near.iloc[:,a2],y_near,marker='x',c='k')
    # Colorbars
    cbaxesr = fig.add_axes([0.88, 0.25, 0.03, 0.5])
    cbr = fig.colorbar(smr, ax=ax, cax=cbaxesr)
    cbr.set_label('Uncertainty', rotation=270, labelpad=20, fontdict={'size': 24})
    cbaxesb = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # This is the position for the colorbar
    cbb = fig.colorbar(smb, ax=ax, cax=cbaxesb, orientation='horizontal')
    cbb.set_label('Utility Function', fontdict={'size': 24})
    cbaxesl = fig.add_axes([0.12, 0.25, 0.03, 0.5])
    cbl = fig.colorbar(smt, ax=ax, cax=cbaxesl, orientation='vertical')
    cbl.set_label('Predicted Mean', rotation=90, fontdict={'size': 24})
    cbaxesl.yaxis.set_ticks_position('left')
    cbaxesl.yaxis.set_label_position('left')
    # Plot observations

    # Plot settings
    vector[a1] = -1
    vector[a2] = -1
    ax.set_xlim((x1_min, x1_max))
    ax.set_ylim((x2_min, x2_max))
    ax.set_zlim((0, y_max * 1.1))
    ax.set_title('Plot along {}'.format(vector), fontdict={'size': 24}, pad=20)
    ax.set_zlabel('Hydrogen generation', fontdict={'size': 24}, labelpad=20)
    ax.set_xlabel('{}'.format(features[a1], vector), fontdict={'size': 24}, labelpad=20)
    ax.set_ylabel('{}'.format(features[a2], vector), fontdict={'size': 24}, labelpad=20)

    if path: plt.savefig(path, dpi=dpi)
    return fig