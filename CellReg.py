from util import open_minian, dir_dict, find_dict_entries
from pathlib import Path
import os
from scipy.io import savemat
import numpy as np
import glob
import h5py
import pickle


class SpatialFootprints():
    def __init__(self, mouse_path):
        """
        Class that handles spatial footprint-related stuff. Currently only
        makes the .mat file that gets fed into Ziv Lab's CellReg Matlab package.
        ***Currently requires that your files are organized like this***:
        -animal
            -session
                -session_id (which contains minian folder)

        :parameter
        ---
        mouse_path: str, animal folder (see  above)

        """
        # Define paths.
        self.mouse_path = mouse_path
        self.session_paths = [folder.parent for folder in Path(self.mouse_path).rglob('minian')]
        self.session_numbers = [folder.parts[-2] for folder in self.session_paths]

    def make_mat(self, save_path=None):
        """
        Makes spatial footprints .mat for CellReg Matlab package.

        :parameter
        ---
        save_path: str, path to save .mat file. Defaults to a folder called
            SpatialFootprints inside the session_id folder.
        """
        if save_path is None:
            save_path = os.path.join(self.mouse_path, 'SpatialFootprints')

        cellreg_path = os.path.join(save_path, 'CellRegResults')
        os.mkdir(save_path)
        os.mkdir(cellreg_path)

        for session, session_number in zip(self.session_paths,
                                           self.session_numbers):
            #File name.
            fname = os.path.join(save_path, session_number+'.mat')

            # Load data.
            data = open_minian(session)

            # Reshape matrix. CellReg reads (neuron, x, y) arrays.
            footprints = np.asarray(data.A)
            footprints = np.rollaxis(footprints, 2)

            # Save.
            savemat(fname,
                    {'footprints': footprints})
            print(f'Saved {fname}')


class CellRegObj:
    def __init__(self, path):
        """
        Object for handling and saving outputs from CellReg Matlab package.

        :parameter
        ---
        path: str, full path to CellRegResults folder.

        """
        self.path = path

        try:
            self.map = self.load_cellreg_results()
        except:
            self.data, self.file  = self.read_cellreg_output()
            self.compile_cellreg_data()
            self.map = self.load_cellreg_results()

    def load_cellreg_results(self, mode='map'):
        """
        After having already running CellRegObj, load the saved pkl file.

        """
        # Get file name based on mode.
        file_dict = {'map': 'CellRegResults.pkl',
                     'footprints': 'CellRegFootprints.pkl',
                     'centroids': 'CellRegCentroids.pkl',
                     }
        fname = os.path.join(self.path, file_dict[mode])

        # Open pkl file.
        with open(fname, 'rb') as file:
            data = pickle.load(file)

        return data

    def read_cellreg_output(self):
        """
        Reads the .mat file.
        :return:
        """
        cellreg_file = glob.glob(os.path.join(self.path,'cellRegistered*.mat'))
        assert len(cellreg_file) > 0, "No registration .mat detected."
        assert len(cellreg_file) is 1, "Multiple cell registration files!"
        cellreg_file = cellreg_file[0]

        # Load it.
        file = h5py.File(cellreg_file)
        data = file['cell_registered_struct']

        return data, file

    def process_registration_map(self):
        # Get the cell_to_index_map. Reading the file transposes the
        # matrix. Transpose it back.
        cell_to_index_map = self.data['cell_to_index_map'].value.T

        # Matlab indexes starting from 1. Correct this.
        match_map = cell_to_index_map - 1

        return match_map.astype(int)

    def process_spatial_footprints(self):
        # Get the spatial footprints after translations.
        footprints_reference = self.data['spatial_footprints_corrected'].value[0]

        footprints = []
        for idx in footprints_reference:
            # Float 32 takes less memory.
            session_footprints = np.float32(np.transpose(self.file[idx].value, (2, 0, 1)))
            footprints.append(session_footprints)

        return footprints

    def process_centroids(self):
        # Also get centroid positions after translations.
        centroids_reference = self.data['centroid_locations_corrected'].value[0]

        centroids = []
        for idx in centroids_reference:
            session_centroids = self.file[idx].value.T
            centroids.append(session_centroids)

        return centroids

    def compile_cellreg_data(self):
        # Gets registration information. So far, this consists of the
        # cell to index map, centroids, and spatial footprints.
        match_map = self.process_registration_map()
        centroids = self.process_centroids()
        footprints = self.process_spatial_footprints()

        filename =\
            os.path.join(self.path,'CellRegResults.pkl')
        filename_footprints = \
            os.path.join(self.path,'CellRegFootprints.pkl')
        filename_centroids = \
            os.path.join(self.path, 'CellRegCentroids.pkl')

        with open(filename, 'wb') as output:
            pickle.dump(match_map, output, protocol=4)
        with open(filename_footprints, 'wb') as output:
            pickle.dump(footprints, output, protocol=4)
        with open(filename_centroids, 'wb') as output:
            pickle.dump(centroids, output, protocol=4)


def trim_map(map, cols, detected='everyday'):
    """
    Eliminates columns in the neuron mapping array.

    :parameters
    ---
    map: (neuron, session) array
        Neuron mappings.

    cols: list-like
        Columns of map to keep.

    detected: str
        'everyday': keeps only the neurons that were detected on
        each session (column) specified.
        'either_day': keeps the neurons that were detected on either
        session specified.

    :return
    ---
    trimmed_map: (neuron, session) array
        Neuron mappings where number of columns is equal to length
        of cols.

    """
    # Take only specified columns.
    trimmed_map = map[:,cols]

    # Eliminate neurons that were not detected on every session.
    if detected == 'everyday':
        neurons_detected = (trimmed_map > -1).all(axis=1)
    elif detected == 'either_day':
        neurons_detected = (trimmed_map > -1).any(axis=1)
    elif detected == 'first_day':
        neurons_detected = trimmed_map[:,0] > -1
    else:
        TypeError('Invalid value for detected')

    trimmed_map = trimmed_map[neurons_detected,:]

    return trimmed_map


def rearrange_neurons(map, neural_data):
    """
    Rearranges (neuron, time) arrays by rows to match the mapping.

    :parameters
    ---
    map: (neuron, session) array
        Neuron mappings.

    neural_data: list of (neuron, time) arrays
        Neural data (e.g., S). List must be in same order as the
        columns in map.

    """
    # Handles cases where only one session was fed in.
    if map.ndim == 1:
        map = np.expand_dims(map, 1)
    neural_data = [neural_data] if not isinstance(neural_data, list) else neural_data

    if not -1 in map:
        # Do a simple rearrangement of rows based on mappings.
        rearranged = []
        for n, session in enumerate(neural_data):
            rearranged.append(session[map[:,n]])

    else:
        # Catches cases where -1s (non-matched neurons) are in map.
        # In this case, iterate through each map column (session)
        # and then iterate through each neuron. Grab it from neural_data
        # if it exists, otherwise, fill it with zeros.
        print('Unmatched neurons detected in map. Padding with zeros')
        rearranged = []

        for n, session in enumerate(neural_data):
            # Assume neuron is not active, fill with zeros.
            rearranged_session = np.zeros((map.shape[0], session.shape[1]))

            # Be sure to only grab neurons if map value is > -1.
            # Otherwise, it will take the last neuron (-1).
            for m, neuron in enumerate(map[:,n]):
                if neuron > -1:
                    rearranged_session[m] = session[neuron]

            # Append rearranged matrix with paddded zeros.
            rearranged.append(rearranged_session)

    return rearranged


def get_cellreg_path(mouse, dict_list=dir_dict(), animal_key = 'Animal',
                     cellreg_key='CellRegPath'):
    """
    Grabs the path containing CellRegResults folder from a dict
    made by dir_dict().

    :parameters
    ---
    mouse: str
        Mouse name.

    dict_list: list of dicts
        From dir_dict(). Should contain a key 'Animal' or something
        similar denoting animal name.

    animal_key: str
        A dict key labeling your animals.

    cellreg_key: str
        A dict key labeling the path to CellRegResults.

    :return
    ---
    path: str
        Path to CellRegResults folder.

    """
    entries = find_dict_entries(dict_list, **{animal_key: mouse})
    path = entries[0][cellreg_key]

    return path

if __name__ == '__main__':
    get_cellreg_path('G132')