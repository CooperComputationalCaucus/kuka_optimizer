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
import os

matplotlib.matplotlib_fname()
matplotlib.font_manager._rebuild()
font = {'size': 18}
matplotlib.rc('font', **font)


def check_dist(gp, a, b, threshold=0.6):
    if gp.kernel_(a.reshape(1, -1), b.reshape(1, -1)) > threshold:
        return True
    else:
        return False


def posterior(gp, grid):
    mu, sigma = gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp_1d(gp, df, axis, vector=None, utility_function=None, path=None, dpi=300, threshold=0.6):
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
    threshold: threshold for kernel simialrity measure

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
        mask[i] = check_dist(gp, np.array(X.iloc[i]), np.array(vector), threshold)
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
    if path: plt.savefig(path, dpi=dpi)

    return fig


def plot_gp_2d(gp, df, a1, a2, vector=None, utility_function=None, path=None, dpi=300, threshold=0.6, n_points=200,
               scatter=False):
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
    threshold: threshold for kernel simialrity measure
    n_points: integer number of points per axis in generating mesh
    scatter: logical to include scatter plot of df data

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
        mask[i] = check_dist(gp, np.array(X.iloc[i]), np.array(vector), threshold)
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

    # Color org
    sig_col = cm.Spectral
    sm_sig = cm.ScalarMappable(cmap=sig_col)
    sm_sig.set_array(sigma_mesh)
    u_col = cm.coolwarm
    sm_u = cm.ScalarMappable(cmap=u_col)
    sm_u.set_array(u_mesh)
    mu_col = cm.PuRd
    sm_mu = cm.ScalarMappable(cmap=mu_col)
    sm_mu.set_array(mu_mesh)

    # Plot prediction
    surf = ax.plot_surface(x1_mesh, x2_mesh, mu_mesh, cmap=mu_col,
                           linewidth=0, antialiased=True, rstride=1, cstride=1, alpha=0.75)
    # Bottom utility
    contb = ax.contourf(x1_mesh, x2_mesh, u_mesh, zdir='z',
                        offset=0, levels=list(np.arange(np.min(utility), np.max(utility), 0.005)), cmap=u_col)
    # Top (bar on left) uncertainty
    contt = ax.contour(x1_mesh, x2_mesh, sigma_mesh, zdir='z',
                       offset=y_max * 1.1, levels=list(np.arange(np.min(sigma), np.max(sigma), 0.05)), cmap=sig_col)

    if scatter:
        ax.scatter(x_near.iloc[:, a1], x_near.iloc[:, a2], marker='.', c='k')

    # Colorbars
    cbaxesr = fig.add_axes([0.88, 0.25, 0.03, 0.5])
    cbr = fig.colorbar(sm_mu, ax=ax, cax=cbaxesr)
    cbr.set_label('Predicted Mean', rotation=270, labelpad=22, fontdict={'size': 24})
    cbaxesb = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # This is the position for the colorbar
    cbb = fig.colorbar(sm_u, ax=ax, cax=cbaxesb, orientation='horizontal')
    cbb.set_label('Utility Function', fontdict={'size': 24})
    cbaxesl = fig.add_axes([0.12, 0.25, 0.03, 0.5])
    cbl = fig.colorbar(sm_sig, ax=ax, cax=cbaxesl, orientation='vertical')
    cbl.set_label('Uncertainty', rotation=90, fontdict={'size': 24})
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


def plot_correlations(df, directory='./', save=False):
    try:
        target_corr = df.corrwith(df.Target)
        f1 = plt.figure(figsize=(20, 8))
        plt.matshow(np.expand_dims(np.array(target_corr[:-1]), axis=0),
                    fignum=f1.number)
        plt.xticks(range(df.shape[1] - 1), df.columns[:-1], rotation=75)
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('Target Correlation', y=3, fontweight="bold");
        plt.show()
        if save:
            f1.savefig(os.path.join(directory, 'target_correlation.png'))
    except AttributeError:
        print('Not set up for working without Target series in DataFrame')

    # 2-d matrix
    cross_correlation = df.corr()
    f2 = plt.figure(figsize=(22, 16))
    plt.matshow(cross_correlation, fignum=f2.number)
    plt.xticks(range(df.shape[1]), df.columns, rotation=75)
    plt.yticks(range(df.shape[1]), df.columns)
    cb = plt.colorbar()
    plt.title('Correlation Matrix', y=1.15, fontweight="bold")
    if save:
        f2.savefig(os.path.join(directory, 'cross_correlation.png'),
                   bbox_inches="tight")
    return f1, f2


def plot_df_3var_2d(df, a1, a2, a3=None, path=None, dpi=300):
    """
    Plots a dataframe sampling along 2 specific axes from complete dataset.
    This can optionally be colored by a third axis.

    Parameters
    ==========
    gp: gaussian process
    df: dataframe of X data with labels, and y data labeled Target
    a1: axis of reference by string or index (abisca)
    a2: second axis of reference by string or index (ordinate)
    a3: third axis of reference by string or index (coloring)
    path: path for plot saving if desired
    dpi: dots per inch for figure output

    Returns
    -------
    fig

    Outputs
    -------
    Figure as png if given path
    """
    # Proccess dataframe
    features = list(df.columns)
    if type(a1) == str:
        a1 = features.index(a1)
    else:
        a1 = int(a1)
    if type(a2) == str:
        a2 = features.index(a2)
    else:
        a2 = int(a2)
    if type(a3) == str:
        a3 = features.index(a3)
        c = df.iloc[:, a3]
    elif a3:
        a3 = int(a3)
        c = df.iloc[:, a3]
    else:
        c = np.zeros_like(a1)

    x = df.iloc[:, a1]
    y = df.iloc[:, a2]

    # Prep Figure
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams['font.sans-serif'] = "DejaVu Sans"
    plt.rcParams['font.family'] = "sans-serif"
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x, y, c=c, cmap=cm.coolwarm)

    # Color bar
    sm = cm.ScalarMappable(cmap=cm.coolwarm)
    sm.set_array(c)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label('{}'.format(features[a3]), fontdict={'size': 16})

    # Plot Settings
    ax.set_xlabel('{}'.format(features[a1]), fontdict={'size': 16})
    ax.set_ylabel('{}'.format(features[a2]), fontdict={'size': 16})

    if path: plt.savefig(path, dpi=dpi)

    return fig


def radar_df(df, exclude=[], normalize=False, path=None, dpi=150, reference_df=None):
    """
    Plots a radar chart of the average value of each dataframe axis
    The optional exlude list can be used to trim down excess axes.
    Automatically excludes target.

    Parameters
    ----------
    df: dataframe of X data with labels, and y data labeled Target
    exclude: list of string labels of axes to exlcude
    path: path for plot saving if desired
    dpi: dots per inch for figure output

    Returns
    -------
    fig

    Outputs
    -------
    Figure as png if given path
    """
    from numpy import pi

    # Collect features
    try:
        df = df.drop(columns=['Target'])
    except:
        pass
    df = df.drop(columns=exclude)
    if reference_df is None:
        features = list(df.columns)
    else:
        try:
            reference_df = reference_df.drop(columns=['Target'])
        except:
            pass
        reference_df = reference_df.drop(columns=exclude)
        _, features = [list(x) for x in zip(*sorted(zip(list(reference_df.mean()), list(reference_df.columns)),
                                                    reverse=True, key=lambda pair: pair[0]))]
        fs = list(df.columns)
    N = len(features)

    # Normalize to 0-1 per axis if requested
    if normalize:
        df = df / df.max()
    # Assemble means as our values to plot
    # Optionally uses reference dataframe to create consistent ordering and axes
    means = list(df.mean())
    if reference_df is None:
        means, features = [list(x) for x in zip(*sorted(zip(means, features), reverse=True, key=lambda pair: pair[0]))]
        yticks = np.arange(0, np.ceil(np.max(means)), 0.5)
    else:
        means = [{f: m for f, m in zip(fs, means)}[feature] for feature in features]
        yticks = np.arange(0, np.ceil(np.max(reference_df.mean())), 0.5)
    means += means[:1]

    # Determine the values of each x axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot and prep figure
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams['font.sans-serif'] = "DejaVu Sans"
    plt.rcParams['font.family'] = "sans-serif"
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable + add labels labels yet
    plt.xticks(angles[:-1], features, color='black', size=16)
    ax.tick_params(axis='x', pad=50)
    # Draw ylabels
    ax.set_rlabel_position(0)

    plt.yticks(yticks, [str(t) for t in yticks], color='grey', size=16)
    plt.ylim(0, np.ceil(np.max(yticks)))
    # Plot data
    ax.plot(angles, means, linewidth=2, linestyle='solid')
    # Fill area
    ax.fill(angles, means, 'b', alpha=0.1)

    plt.tight_layout()
    if path: plt.savefig(path, dpi=dpi)

    return fig


def radar_dfs(dfs, exclude=[], overlay=True, normalize=False, figsize=(10, 10), path=None, dpi=150):
    """
    Plots a radar chart of the average value of each dataframe axis
    The optional exlude list can be used to trim down excess axes.
    Automatically excludes target.

    Depends on the first df in the list for axis sizing and labeling

    Parameters
    ----------
    figsize: optional tuple for figsize
    normalize: Logical to normalize to max of each axis
    overlay: Logical to overlay multiple plots or create subplots
    dfs: list of dataframes of X data with labels
    exclude: list of string labels of axes to exlcude
    path: path for plot saving if desired
    dpi: dots per inch for figure output

    Returns
    -------
    fig

    Outputs
    -------
    Figure as png if given path
    """
    from numpy import pi
    # Preprocessing
    for i in range(len(dfs)):
        try:
            dfs[i] = dfs[i].drop(columns=['Target'])
        except:
            pass
        dfs[i] = dfs[i].drop(columns=exclude)
        # Normalize to 0-1 per axis if requested
        if normalize:
            dfs[i] = dfs[i] / dfs[i].max()

    # Determine the values of each x axis, and sorting from last df
    features = list(dfs[-1].columns)
    N = len(features)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    means = list(dfs[-1].mean())
    means, features = [list(x) for x in zip(*sorted(zip(means, features), reverse=True, key=lambda pair: pair[0]))]
    means += means[:1]

    # Prep figure
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = "DejaVu Sans"
    plt.rcParams['font.family'] = "sans-serif"

    if overlay:
        ax = plt.subplot(111, polar=True)
        # Draw one axis per variable + add labels labels yet
        plt.xticks(angles[:-1], features, size=16)
        ax.tick_params(axis='x', pad=50)
        # Draw ylabels
        ax.set_rlabel_position(0)
        yticks = np.arange(0, np.ceil(np.max(means)), 0.5)
        plt.yticks(yticks, [str(t) for t in yticks], color="grey", size=16)
        plt.ylim(0, np.ceil(np.max(means)))
        # Assembe and add plots
        for df in dfs:
            means = list(df.mean())
            fs = list(df.columns)
            means = [{f: m for f, m in zip(fs, means)}[feature] for feature in features]
            means += means[:1]
            n_samples = len(df)
            ax.plot(angles, means, linewidth=2, linestyle='solid', label='{} Samples'.format(n_samples))
            ax.fill(angles, means, alpha=0.1)
        ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=12)
    else:
        raise Exception("Yet to implement split sublplots for this function")
    plt.tight_layout()
    if path: plt.savefig(path, dpi=dpi)

    return fig


def radar_dfs_gif(dfs, path='./radar.gif', duration=20, **kwargs):
    import os
    import shutil

    def make_gif(filenames, out_path, _duration=20):
        import imageio
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_path, images, 'GIF', duration=_duration)

    def make_mp4(filenames, out_path, _duration=20):
        import imageio
        # images = []
        fps = len(filenames)/_duration
        videowriter = imageio.get_writer(out_path, fps=fps)
        for filename in filenames:
            videowriter.append_data(imageio.imread(filename))
        videowriter.close()

    os.makedirs('./tmp',exist_ok=True)

    paths = []
    for idx, df in enumerate(dfs):
        _path = './tmp/{}.png'.format(idx)
        paths.append(_path)
        f=radar_df(df, path=_path, reference_df=dfs[-1], **kwargs)
        plt.close(f)
    make_gif(paths, path, _duration=duration)
    make_mp4(paths, os.path.splitext(path)[0]+'.mp4')
    shutil.rmtree('./tmp')





