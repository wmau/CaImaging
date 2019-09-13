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
from util import open_minian, get_transient_timestamps
from CellReg import CellRegObj, trim_map, rearrange_neurons
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
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
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
    patterns, significance, z_data = \
        runPatterns(neural_data, method=method, nullhyp=nullhyp,
                    nshu=n_shuffles, percentile=percentile,
                    tracywidom=tracywidow)

    if compute_activity:
        activations = computeAssemblyActivity(patterns, z_data)
    else:
        activations = None

    if plot:
        plt.plot(activations.T)

    assembly_dict = {'patterns':        patterns,
                     'significance':    significance,
                     'z_data':          z_data,
                     'orig_data':       neural_data,
                     'activations':     activations,
                     }

    return assembly_dict


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
    n_sessions = len(lapsed_data) + 1
    lapsed_data = [lapsed_data] if not isinstance(lapsed_data, list) else lapsed_data

    # For cases where you want to specify which neurons to consider for some
    # reason.
    if neurons is not None:
        template_data = template_data[neurons]

    # Get patterns.
    patterns, significance, z_data = \
        runPatterns(template_data, method=method, nullhyp=nullhyp,
                    nshu=n_shuffles, percentile=percentile)

    # Find assembly activations for the template session then the lapsed ones.
    activations = []
    activations.append(computeAssemblyActivity(patterns,
                                               gaussian_filter1d(z_data, 2, 1)))
    for session in lapsed_data:
        z_session = stats.zscore(session, axis=1)
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=0)
        z_session = imp.fit_transform(z_session.T).T
        z_session = gaussian_filter1d(z_session, 2, 1)

        activations.append(computeAssemblyActivity(patterns, z_session))

    # Get event timestamps.
    spiking, rate = get_transient_timestamps(template_data)
    spiking, rate = [spiking], [rate]
    for session in lapsed_data:
        temp_s, temp_r = get_transient_timestamps(session)
        spiking.append(temp_s)
        rate.append(temp_r)

    # Plot assembly activations.
    if plot:
        plot_assemblies(activations, spiking)

        plt.show()

    hub_neurons = get_important_neurons(patterns)

    pass


def plot_assemblies(assembly_act, spiking, do_zscore=True):
    n_sessions = len(assembly_act)
    fig, axes = plt.subplots(n_sessions, 1)

    for n, (ax, act, spikes) in enumerate(zip(axes, assembly_act, spiking)):
        if do_zscore:
            act = stats.zscore(act.T, axis=0)

        ax.plot(act)
        ax2 = ax.twinx()

        ylims = ax.get_ylim()
        ax.set_ylim(bottom=0)
        offsets = np.sum(np.abs(ylims))
        ax2.eventplot(spikes)
        ax2.set_ylim(bottom=0)


def get_important_neurons(patterns, mode='raw', n=10):

    if mode == 'percentile':
        n = (100-n) * patterns.shape[1]

    inds = []
    for pattern in patterns:
        inds.append(np.argpartition(pattern, -n)[-n:])

    # i = 2
    # plt.stem(range(480), patterns[i])
    # plt.stem(hub_neurons[i], patterns[i][hub_neurons[i]], markerfmt='ro')

    return inds

if __name__ == '__main__':
    dpath1 = r'D:\Projects\GTime\Data\G132\1\H11_M0_S21'
    dpath2 = r'D:\Projects\GTime\Data\G132\2\H15_M43_S22'
    dpath8 = r'D:\Projects\GTime\Data\G132\8\H11_M5_S56 - G132.1'
    cellregpath = r'D:\Projects\GTime\Data\G132\SpatialFootprints\CellRegResults'
    minian1 = open_minian(dpath1)
    minian2 = open_minian(dpath2)
    minian8 = open_minian(dpath8)
    C = CellRegObj(cellregpath)


    # map = trim_map(C.map, [0,1], was_detected_everyday=True)
    # template = np.asarray(minian1.S)
    # lapsed = rearrange_neurons(map[:,1], [np.asarray(minian2.S)])
    # template = rearrange_neurons(map[:,0], [template])
    #
    # lapsed_activation(template[0], [lapsed])

    map = trim_map(C.map, [0,1], detected='everyday')
    template = np.asarray(minian1.S)
    lapsed = rearrange_neurons(map[:,1], [np.asarray(minian2.S)])
    template = rearrange_neurons(map[:,0], [template])

    lapsed_activation(template[0], lapsed)