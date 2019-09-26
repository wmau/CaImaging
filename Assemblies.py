'''
	Codes for PCA/ICA methods described in Detecting cell assemblies in large neuronal populations, Lopes-dos-Santos et al (2013).
											https://doi.org/10.1016/j.jneumeth.2013.04.010
	This implementation was written in Feb 2019.
	Please e-mail me if you have comments, doubts, bug reports or criticism (Vítor, vtlsantos@gmail.com /  vitor.lopesdossantos@pharm.ox.ac.uk).
'''

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import stats
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

#### Custom imports start here ####
from util import open_minian, get_transient_timestamps, distinct_colors, \
    ordered_unique, find_dict_entries
import util
from itertools import zip_longest


from CellReg import CellRegObj, trim_map, rearrange_neurons, get_cellreg_path
from scipy.ndimage import gaussian_filter1d


__author__ = "Vítor Lopes dos Santos"
__version__ = "2019.1"


def toyExample(assemblies, nneurons=10, nbins=1000, rate=1.):
    np.random.seed()

    actmat = np.random.poisson(rate, nneurons * nbins).reshape(nneurons, nbins)
    assemblies.actbins = [None] * len(assemblies.membership)
    for (ai, members) in enumerate(assemblies.membership):
        members = np.array(members)
        nact = int(nbins * assemblies.actrate[ai])
        actstrength_ = rate * assemblies.actstrength[ai]

        actbins = np.argsort(np.random.rand(nbins))[0:nact]

        actmat[members.reshape(-1, 1), actbins] = np.ones((len(members), nact)) + actstrength_

        assemblies.actbins[ai] = np.sort(actbins)

    return actmat


class toyassemblies:

    def __init__(self, membership, actrate, actstrength):
        self.membership = membership
        self.actrate = actrate
        self.actstrength = actstrength


def marcenkopastur(significance):
    nbins = significance.nbins
    nneurons = significance.nneurons
    tracywidom = significance.tracywidom

    # calculates statistical threshold from Marcenko-Pastur distribution
    q = float(nbins) / float(nneurons)  # note that silent neurons are counted too
    lambdaMax = pow((1 + np.sqrt(1 / q)), 2)
    lambdaMax += tracywidom * pow(nneurons, -2. / 3)  # Tracy-Widom correction

    return lambdaMax


def getlambdacontrol(zactmat_):
    significance_ = PCA()
    significance_.fit(zactmat_.T)
    lambdamax_ = np.max(significance_.explained_variance_)

    return lambdamax_


def binshuffling(zactmat, significance):
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            randomorder = np.argsort(np.random.rand(significance.nbins))
            zactmat_[neuroni, :] = activity[randomorder]
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def circshuffling(zactmat, significance):
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            cut = int(np.random.randint(significance.nbins * 2))
            zactmat_[neuroni, :] = np.roll(activity, cut)
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def runSignificance(zactmat, significance):
    if significance.nullhyp == 'mp':
        lambdaMax = marcenkopastur(significance)
    elif significance.nullhyp == 'bin':
        lambdaMax = binshuffling(zactmat, significance)
    elif significance.nullhyp == 'circ':
        lambdaMax = circshuffling(zactmat, significance)
    else:
        print('ERROR !')
        print('    nyll hypothesis method ' + str(nullhyp) + ' not understood')
        significance.nassemblies = np.nan

    nassemblies = np.sum(significance.explained_variance_ > lambdaMax)
    significance.nassemblies = nassemblies

    return significance


def extractPatterns(actmat, significance, method):
    nassemblies = significance.nassemblies

    if method == 'pca':
        idxs = np.argsort(-significance.explained_variance_)[0:nassemblies]
        patterns = significance.components_[idxs, :]
    elif method == 'ica':
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=nassemblies)
        ica.fit(actmat.T)
        patterns = ica.components_
    else:
        print('ERROR !')
        print('    assembly extraction method ' + str(method) + ' not understood')
        patterns = np.nan

    if patterns is not np.nan:
        patterns = patterns.reshape(nassemblies, -1)

        # sets norm of assembly vectors to 1
        norms = np.linalg.norm(patterns, axis=1)
        patterns /= np.matlib.repmat(norms, np.size(patterns, 1), 1).T

    return patterns


def runPatterns(actmat, method='ica', nullhyp='circ', nshu=1000, percentile=99, tracywidom=False):
    '''
    INPUTS

        actmat:     activity matrix - numpy array (neurons, time bins)

        nullhyp:    defines how to generate statistical threshold for assembly detection.
                        'bin' - bin shuffling, will shuffle time bins of each neuron independently
                        'circ' - circular shuffling, will shift time bins of each neuron independently
                                                            obs: mantains (virtually) autocorrelations
                        'mp' - Marcenko-Pastur distribution - analytical threshold

        nshu:       defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

        percentile: defines which percentile to be used use when shuffling methods are employed.
                                                                    (n/a if nullhyp is 'mp')

        tracywidow: determines if Tracy-Widom is used. See Peyrache et al 2010.
                                                (n/a if nullhyp is NOT 'mp')

    OUTPUTS

        patterns:     co-activation patterns (assemblies) - numpy array (assemblies, neurons)
        significance: object containing general information about significance tests
        zactmat:      returns z-scored actmat

    '''

    nneurons = np.size(actmat, 0)
    nbins = np.size(actmat, 1)

    silentneurons = np.var(actmat, axis=1) == 0
    actmat_ = actmat[~silentneurons, :]

    # z-scoring activity matrix
    zactmat_ = stats.zscore(actmat_, axis=1)

    # Impute missing values.
    imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                        fill_value=0)
    zactmat_ = imp.fit_transform(zactmat_.T).T

    # running significance (estimating number of assemblies)
    significance = PCA()
    significance.fit(zactmat_.T)
    significance.nneurons = nneurons
    significance.nbins = nbins
    significance.nshu = nshu
    significance.percentile = percentile
    significance.tracywidom = tracywidom
    significance.nullhyp = nullhyp
    significance = runSignificance(zactmat_, significance)
    if np.isnan(significance.nassemblies):
        return

    if significance.nassemblies < 1:
        print('WARNING !')
        print('    no assembly detecded!')
        patterns = []
    else:
        # extracting co-activation patterns
        patterns_ = extractPatterns(zactmat_, significance, method)
        if patterns_ is np.nan:
            return

        # putting eventual silent neurons back (their assembly weights are defined as zero)
        patterns = np.zeros((np.size(patterns_, 0), nneurons))
        patterns[:, ~silentneurons] = patterns_
        zactmat = np.copy(actmat)
        zactmat[~silentneurons, :] = zactmat_


    return patterns, significance, zactmat


def computeAssemblyActivity(patterns, zactmat, zerodiag=True):
    nassemblies = len(patterns)
    nbins = np.size(zactmat, 1)

    assemblyAct = np.zeros((nassemblies, nbins))
    for (assemblyi, pattern) in enumerate(patterns):
        projMat = np.outer(pattern, pattern)
        projMat -= zerodiag * np.diag(np.diag(projMat))
        for bini in range(nbins):
            assemblyAct[assemblyi, bini] = \
                np.dot(np.dot(zactmat[:, bini], projMat), zactmat[:, bini])

    return assemblyAct

################### Will's code starts here ##################

def find_assemblies(neural_data, method='ica', nullhyp='mp', n_shuffles=1000,
                    percentile=99, tracywidow=False, compute_activity=True, plot=True):
    """
    Gets patterns and assembly activations in one go.

    :parameters
    ---
    neural_data: (neuron, time) array
        Neural activity (e.g., S).

    method: str
        'ica' or 'pca'. 'ica' is recommended.

    nullhyp: str
        defines how to generate statistical threshold for assembly detection.
            'bin' - bin shuffling, will shuffle time bins of each neuron independently
            'circ' - circular shuffling, will shift time bins of each neuron independently
                     obs: maintains (virtually) autocorrelations
             'mp' - Marcenko-Pastur distribution - analytical threshold

    nshu: float
        defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

    percentile: float
        defines which percentile to be used use when shuffling methods are employed.
        (n/a if nullhyp is 'mp')

    tracywidow: bool
        determines if Tracy-Widom is used. See Peyrache et al 2010.
        (n/a if nullhyp is NOT 'mp')

    """
    spiking, _, bool_arr = util.get_transient_timestamps(neural_data)

    patterns, significance, z_data = \
        runPatterns(bool_arr, method=method, nullhyp=nullhyp,
                    nshu=n_shuffles, percentile=percentile,
                    tracywidom=tracywidow)

    if compute_activity:
        activations = computeAssemblyActivity(patterns, bool_arr)

        if plot:
            sorted_spiking, sorted_colors = membership_sort(patterns, spiking)
            plot_assemblies(activations, sorted_spiking, colors=sorted_colors)
    else:
        activations = None

    assembly_dict = {'patterns':        patterns,
                     'significance':    significance,
                     'z_data':          z_data,
                     'orig_data':       neural_data,
                     'activations':     activations,
                     }

    return assembly_dict


def membership_sort(patterns, neural_data, sort_duplicates=True):
    """
    Sorts neurons by their contributions to each pattern.

    :param patterns:
    :param neural_data:
    :return:
    """
    high_weights = get_important_neurons(patterns)
    colors = util.distinct_colors(patterns.shape[0])

    do_not_sort, sorted_data, sorted_colors = [], [], []
    for color, pattern in zip(colors, high_weights):
        for neuron in pattern:
            if neuron not in do_not_sort:
                sorted_data.append(neural_data[neuron])
                sorted_colors.append(color)

                if not sort_duplicates:
                    do_not_sort.append(neuron)

    return sorted_data, sorted_colors



def lapsed_activation(template_data, lapsed_data,  method='ica',
                      nullhyp='mp', n_shuffles=1000, percentile=99,
                      plot=True, neurons=None):
    """
    Computes activity of ensembles based on data from another day.

    :parameters
    ---
    map: (neurons, sessions) array.
        Mapping of neuron IDs across sessions.

    template_data: (neurons, time) array.
        Neural activity.

    lapsed_data: list of (neurons, time) arrays. len of list must be sessions-1
        Neural activity from other sessions.
    """
    # Useful variables.
    lapsed_data = [lapsed_data] if not isinstance(lapsed_data, list) else lapsed_data

    # For cases where you want to specify which neurons to consider for some
    # reason.
    if neurons is not None:
        template_data = template_data[neurons]

    # Get event timestamps.
    spiking, rate, bool_arr = util.get_transient_timestamps(template_data)
    spiking, rate, bool_arr = [spiking], [rate], [bool_arr]
    for session in lapsed_data:
        temp_s, temp_r, temp_bool = util.get_transient_timestamps(session)
        spiking.append(temp_s)
        rate.append(temp_r)
        bool_arr.append(temp_bool)

    # Get patterns.
    patterns, significance, z_data = \
        runPatterns(bool_arr[0], method=method, nullhyp=nullhyp,
                    nshu=n_shuffles, percentile=percentile)

    # Find assembly activations for the template session then the lapsed ones.
    activations = []
    # bool_arr[0] used to be z_data
    activations.append(computeAssemblyActivity(patterns, bool_arr[0]))
    for session in bool_arr[1:]:
        # Handle missing data.
        z_session = stats.zscore(session, axis=1)
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=0)
        z_session = imp.fit_transform(z_session.T).T
        #z_session = gaussian_filter1d(z_session, 2, 1)

        # Get activations.
        activations.append(computeAssemblyActivity(patterns, z_session))


    # Sort neurons based on membership (weights) in different patterns.
    sorted_spikes, color_list = [], []
    for session in spiking:
        session_sorted, colors_sorted = membership_sort(patterns, session)

        # Do this for each session.
        color_list.append(colors_sorted)
        sorted_spikes.append(session_sorted)


    # Plot assembly activations.
    if plot:
        fig, axes = plot_assemblies(activations, sorted_spikes, colors=color_list)

        plt.tight_layout()
        plt.show()
    else:
        fig, axes = None, None

    return activations, patterns, sorted_spikes, fig, axes


def plot_assemblies(assembly_act, spiking, do_zscore=True, colors=None):
    """
    Plots assembly activations with spikes overlaid.

    :parameters
    ---
    assembly_act: list of (patterns, time) arrays
        Assembly activations.

    spiking: (sessions,) list of (neurons,) lists
        The inner lists should contain timestamps of spiking activity (e.g., from S).

    do_zscore: bool
        Flag to z-score assembly_act.

    colors: (sessions,) list of (neurons,) lists
        The inner lists should contain colors for each neuron.

    """

    # Handles cases where you only want to plot one session's assembly.
    if not isinstance(assembly_act, list):
        assembly_act = [assembly_act]

        # If colors are not specified, use defaults.
        if colors is None:
            colors = util.distinct_colors(assembly_act[0])

        # spiking should already be a list. Let's also check that it's a list
        # that's the same size as assembly_act. If not, it's probably a list
        # of a single session so package it into a list.
        if len(spiking) != len(assembly_act):
            spiking = [spiking]
            colors = [colors]

    # Get color for each assembly.
    uniq_colors = util.ordered_unique(colors[0])

    # Build the figure.
    n_sessions = len(assembly_act)
    fig, axes = plt.subplots(n_sessions, 1)
    if n_sessions == 1:
        axes = [axes]       # For iteration purposes.

    # For each session, plot each assembly.
    for n, (ax, act, spikes, c) in \
            enumerate(zip_longest(axes, assembly_act, spiking, colors,
                                  fillvalue='k')):
        if do_zscore:
            act = stats.zscore(act, axis=1)

        # Plot assembly activation.
        for pattern, pattern_color in zip(act, uniq_colors):
            ax.plot(pattern, color=pattern_color, alpha=0.7)
        ax2 = ax.twinx()

        # Plot spikes.
        ax2.eventplot(spikes, colors=c)
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

    return fig, axes


def get_important_neurons(patterns, mode='raw', n=10):
    """
    Gets the most highly contributing neurons from each pattern.

    :parameters
    ---
    patterns: (patterns, neurons) array
        Weights for each neuron.

    mode: 'raw' or 'percentile'
        Determines whether to interpret n as a percentile or the raw number.

    n: float
        Percentile or number of neurons to extract from pattern weightings.

    :return
    ---
    inds: (patterns,) list of (n,) arrays
        Neuron indices.

    """
    if mode == 'percentile':
        n = (100-n) * patterns.shape[1]

    inds = []
    for pattern in np.abs(patterns):
        inds.append(np.argpartition(pattern, -n)[-n:])

    return inds

if __name__ == '__main__':
    s1 = 0
    s2 = 2
    mouse = 'G132'
    dict_list = util.dir_dict()
    entries = util.find_dict_entries(dict_list, **{'Animal': mouse})
    cellregpath = get_cellreg_path(mouse)

    minian_outputs = []
    for entry in entries:
        minian_outputs.append(open_minian(entry['DataPath']))

    C = CellRegObj(cellregpath)


    # map = trim_map(C.map, [0,1], was_detected_everyday=True)
    # template = np.asarray(minian1.S)
    # lapsed = rearrange_neurons(map[:,1], [np.asarray(minian2.S)])
    # template = rearrange_neurons(map[:,0], [template])
    #
    # lapsed_activation(template[0], [lapsed])

    map = trim_map(C.map, [s1,s2], detected='either_day')
    template = np.asarray(minian_outputs[s1].S)
    lapsed = rearrange_neurons(map[:,1], [np.asarray(minian_outputs[s2].S)])
    template = rearrange_neurons(map[:,0], [template])

    lapsed_activation(template[0], lapsed)
    #find_assemblies(template[0])