import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
from typing import Self
from pprint import pprint
from IPython.display import display, Markdown

import qutip
import pickle



def local_unitary_QWP(theta):
    return (1/2) * np.array([[1 + 1j, (1 - 1j) * np.exp(2j * theta)], [(1 - 1j) * np.exp(-2j * theta), 1 + 1j]])
def local_unitary_HWP(theta):
    return np.array([[0, np.exp(2j * theta)], [np.exp(-2j * theta), 0]])
def local_polarization_unitary(theta_Q, theta_H):
    return np.dot(local_unitary_QWP(theta_Q), local_unitary_HWP(theta_H))

def filter_all_data_rows(all_data, rows_specs):
    # filter the rows of the all_data dataframe
    # if `rows` is a string, it's used to decide among predefined rows
    if isinstance(rows_specs, str):
        if rows_specs == 'sep1':
            all_data = all_data[all_data['label'] == 'sep1']
        elif rows_specs == 'sep2':
            all_data = all_data[all_data['label'] == 'sep2']
        elif rows_specs == 'allsep':
            all_data = all_data[all_data['label'].str.contains('sep')]
        elif rows_specs == 'ent':
            all_data = all_data[all_data['label'] == 'ent']
    else:
        # if not a string, it's assumed to be a dataframe with bool values
        # to be used as a mask for the all_data dataframe
        all_data = all_data[rows_specs]
    return all_data

# function to load all pickle files from a given directory
def load_pickle_files(directory):
    pickle_files = []
    for file in os.listdir(directory):
        if file.endswith('.pickle'):
            with open(os.path.join(directory, file), 'rb') as f:
                pickle_files.append((file, pickle.load(f)))
    return pickle_files

# function to load raw data from the pickle files (the pickles in "Ripetizione_X/Entangled|Separabili/Raw/YYY.pickle")
def load_raw_counts_from_files(directory):
    # the given directory should be the one containing the "Ripetizione_X" folders
    pickle_files = load_pickle_files(directory)
    # output data is in the form of a pandas dataframe. Each element of the dataframe contains the following:
    # ---- state_id, repetition_number, singles_1, singles_2, doubles
    new_data = []
    for file, data in pickle_files:
        df = pd.json_normalize(data['Raw_data'])
        # extract the singles and put them in two matrices, one per walker
        singles_1 = np.asarray([row[0] for row in df['singles']])  # gives the singles as a list of length 25
        singles_2 = np.asarray([row[1] for row in df['singles']])  # gives the singles as a list of length 25
        # the variable file has structure like Dataset_Quantum_stato_108_ent_0_ripetizione_0_2024,09,20-15,25,49_raw.pickle
        # extract the state number and the repetition number from the file name
        state_id = int(file.split('_')[3])
        repetition_number = int(file.split('_')[7])
        # extract info about the doubles
        doubles = np.asarray(df['doubles'])
        new_data.append({
            'state_id': state_id, 'rep': repetition_number,
            'singles_1': singles_1, 'singles_2': singles_2, 'doubles': doubles
            })
    return pd.DataFrame(new_data)

def load_angles_from_file(path: str, filename: str):
    """Load angles from the `Angoli_*WPX_type.txt` files.
    
    This is used to take a single file containing a list of angles for a single
    type of waveplate (HWP or QWP), and return a numpy array containing the angles.

    This is NOT to load the angles in the raw data folders.

    NOTE: The angles in the files are assumed to be in degrees,
          and are here converted to radians.
    """
    # the file contains a list of real numbers, either comma separated or newline separated
    with open(os.path.join(path, filename), 'r') as f:
        lines = f.read().replace(',', '\n').split()
        lines = np.asarray([float(line) for line in lines])
        lines = lines * np.pi / 180  # convert to radians
        return lines

def load_angles_from_raw_file(fullpath: str) -> dict:
    """Load the angles from a single raw data file.
    
    Each of these files is a dataframe, with columns: 'State', 'Angle_pump', 'Angle_X', and others.
    Each has 5 rows.
    Most columns contain a constant value, with the exception of the columns for doubles (which we ignore here).
    """
    with open(fullpath, 'rb') as f:
        data = pickle.load(f)
    data_dict = {}
    data_dict['state_id'] = data['State'][0]
    # extract angles data (converting from degrees to radians)
    for col in data.columns:
        if col.startswith('Angle_') and col != 'Angle_pump':
            data_dict[col[6:]] = data[col][0] * np.pi / 180
    # if there's an 'Angle_pump' column, use it to determine label (this happens in the 2024-09-30 dataset)
    # I need the exception for the 2025-01-09 dataset b/c there we have Angle_pump (set to 5 for separables), but only one reference state
    if 'Angle_pump' in data.columns:
        # save angle pump (corresponds to label)
        if data['Angle_pump'][0] == 5:
            data_dict['label'] = 'sep'
            # why is Angle_pump=5 and not 0 for separable states, in the 2025-01-09 dataset? NO IDEA!
        if data['Angle_pump'][0] == 0:
            data_dict['label'] = 'sep1'
        elif data['Angle_pump'][0] == 45:
            data_dict['label'] = 'sep2'
        elif data['Angle_pump'][0] == 27:
            data_dict['label'] = 'ent'
    # if there's no 'Angle_pump' column, infer ent or sep from the filename
    else:
        if 'Separabili' in fullpath:
            data_dict['label'] = 'sep'
        elif 'Entangled' in fullpath:
            data_dict['label'] = 'ent'
        else:
            raise ValueError('Could not determine label from the filename')
    return data_dict

def load_angles_from_raw_files(directory: str) -> pd.DataFrame:
    """Load the angles from all the raw (angle) files in a given directory.
    
    This function loads the angles from the raw data files, which are in the form of pickles.
    Each pickle contains a dataframe with the angles for each state.

    The output is a pandas dataframe with the columns: 'state_id', 'QWP', 'HWP', 'label'.
    """
    # list all pickle files in the directory
    pickle_files = [file for file in os.listdir(directory) if file.endswith('.pickle')]
    # load the angles from each file
    angles_data = []
    for file in pickle_files:
        angles_data.append(load_angles_from_raw_file(os.path.join(directory, file)))
    return pd.DataFrame(angles_data)

class ExperimentalDataset:
    """Class to handle the data from the experimental datasets.
    
    This class is used to load the data from the experimental datasets.
    It can load the data from the raw data files, or from the txt files.
    The data is stored in the class attributes, and can be accessed using the appropriate methods.

    Attributes
    ----------
        path : str
            The path to the dataset.
        dataset_name : str
            The name of the dataset.
        date : str
            The date of the dataset (eg 2024-09-30, 2024-09-20, etc).
        states_data : pd.DataFrame
            The dataframe containing the states for each label.
        counts : np.ndarray
            The counts (doubles) for each state.
            Note that this connects with the elements of states_data via the state_id values (loaded from the raw data file names).
            In most datasets, each state is uniquely identified by its state id, which makes this easy.
            
            The 2025-01-09 dataset is an exception because there the state ids go from 0 to 149 for both separable and entangled states.
            In this case we thus manually add 150 to the state ids of entangled states, to make them unique.
        singles_1 : np.ndarray
            The singles for the first walker (only if raw data is available).
        singles_2 : np.ndarray
            The singles for the second walker (only if raw data is available).
        labels : list[str]
            The labels for the states (sep, ent, sep1, sep2).
        angles_labels : list[str]
            The labels for the angles data (QWP, QWP1, etc).


    """
    def __init__(self, path: str, force_reload: bool = False, stfu: bool = True) -> None:
        """Initialise the dataset object.
        """
        self.path = path
        # extract the last bit from the given path
        self.dataset_name = os.path.basename(path)
        # if it has the form 'dati dd-mm', extract the date, otherwise throw an error
        if self.dataset_name.startswith('dati '):
            self.date = self.dataset_name.split(' ')[1]
        else:
            raise ValueError('The given path does not have the form "dati dd-mm"')
        # log
        self.stfu = stfu
        if not stfu:
            display(Markdown(f'**Loading dataset**: \'*{self.dataset_name}*\''))

        # store labels corresponding to each set of states (one label per reference input state)
        # and also define the corresponding reference states (these are used to compute the input states from the angles)
        # ---------- NOTE: CHECK THAT THE REFERENCE STATE FOR 2024-11-08 IS CORRECT ------------
        if self.date in ['2024-09-20', '2024-09-02', '2024-11-08', '2025-01-09']:
            self.labels = ['sep', 'ent']
            self.reference_states = {
                'sep': qutip.Qobj([1, 1, -1, -1], dims=[[2, 2], [1, 1]]).unit(),
                'ent': qutip.Qobj([1, 0, 0, -1], dims=[[2, 2], [1, 1]]).unit()
            }
        elif self.date in ['2024-09-30']:
            self.labels = ['sep1', 'sep2', 'ent']
            self.reference_states = {
                'sep1': qutip.Qobj([1, 1, -1, -1], dims=[[2, 2], [1, 1]]).unit(),
                'sep2': qutip.Qobj([1, -1, 1, -1], dims=[[2, 2], [1, 1]]).unit(),
                'ent': qutip.Qobj([0, 1, -1, 0], dims=[[2, 2], [1, 1]]).unit()
            }

        # if an all_data file exists, we just use that one:
        if 'all_data.pickle' in os.listdir(self.path) and not force_reload:
            if not stfu:
                display(Markdown(f'**Loading data from** *`{self.path}\\all_data.pickle`*'))

            with open(os.path.join(self.path, 'all_data.pickle'), 'rb') as f:
                all_data = pickle.load(f)
                self.states_data = all_data[0]
                self.counts = all_data[1]
                if len(all_data) > 2:
                    self.singles_1 = all_data[2]
                    self.singles_2 = all_data[3]
        elif 'Ripetizione_0' in os.listdir(self.path):
            if not stfu:
                display(Markdown(f'***Raw data folders detected. Loading from there...***'))
            # states_data is first created here
            self._load_angles_data_from_raw_files()
            if not stfu:
                display(Markdown(f'***Generate states from angles...***'))
            self._load_states()
            if not stfu:
                display(Markdown(f'***Loading raw counts...***'))
            self._load_raw_counts()

        else:
            if not stfu:
                display(Markdown(f'***`all_data.pickle` not found, and no raw data found. Loading from txt files...***'))
            # this populates the angles dictionary with the angles data
            self._load_angles_data()
            # now compute the input states using the angles
            self._load_states()
            # now load the counts
            self._load_counts()
       

    
    def _find_angles_files(self):
        acceptable_filenames = [
            'Angoli_QWP_ent.txt', 'Angoli_QWP_sep.txt',
            'Angoli_QWP1_ent.txt', 'Angoli_QWP1_sep.txt',
            'Angoli_QWP2_ent.txt', 'Angoli_QWP2_sep.txt',
            'Angoli_HWP_ent.txt', 'Angoli_HWP_sep.txt',
            'Angoli_HWP1_ent.txt', 'Angoli_HWP1_sep.txt',
            'Angoli_HWP2_ent.txt', 'Angoli_HWP2_sep.txt'
        ]
        self.angles_datafiles = [file for file in os.listdir(self.path) if file in acceptable_filenames]

    
    def _load_angles_data(self) -> Self:
        """Load the angles data from the txt files.
        
        This also populates self.angles_labels with the labels for the angles data.
        The generated dataframe is stored in self.states_data (which will later be also populated with the states).
        """
        self._find_angles_files()
        # print the angles files found
        display(Markdown(f'**Angles files found**: {self.angles_datafiles}'))
        # load the angles data from the files
        if self.date in ['2024-09-20', '2024-09-02']:
            self.angles_labels = ['QWP', 'HWP']
            # if the date is either 2024-09-20 or 2024-09-02, the two walker use the same angle
            df_sep = pd.DataFrame({
                'QWP': load_angles_from_file(self.path, 'Angoli_QWP_sep.txt'),
                'HWP': load_angles_from_file(self.path, 'Angoli_HWP_sep.txt')
            })
            # add a label column to the dataframe whose value alternates between 'sep1' and 'sep2'
            df_sep['label'] = 'sep'
            df_ent = pd.DataFrame({
                'QWP': load_angles_from_file(self.path, 'Angoli_QWP_ent.txt'),
                'HWP': load_angles_from_file(self.path, 'Angoli_HWP_ent.txt'),
            })
            df_ent['label'] = 'ent'

        elif self.date in ['2024-09-30']:
            self.angles_labels = ['QWP1', 'QWP2', 'HWP1', 'HWP2']
            # 2024-09-30 uses different angles for the two walkers
            # NOTE: here we also have two possible reference separable states
            #       so each angle for separable states appears twice, once per reference state.
            #       We mark this using as labels 'sep1' and 'sep2', corresponding to the two reference states
            df_sep = pd.DataFrame({
                'QWP1': load_angles_from_file(self.path, 'Angoli_QWP1_sep.txt'),
                'QWP2': load_angles_from_file(self.path, 'Angoli_QWP2_sep.txt'),
                'HWP1': load_angles_from_file(self.path, 'Angoli_HWP1_sep.txt'),
                'HWP2': load_angles_from_file(self.path, 'Angoli_HWP2_sep.txt')
            })
            # add a label column to the dataframe whose value alternates between 'sep1' and 'sep2'
            df_sep['label'] = ['sep1', 'sep2'] * (df_sep.shape[0] // 2)
            df_ent = pd.DataFrame({
                'QWP1': load_angles_from_file(self.path, 'Angoli_QWP1_ent.txt'),
                'QWP2': load_angles_from_file(self.path, 'Angoli_QWP2_ent.txt'),
                'HWP1': load_angles_from_file(self.path, 'Angoli_HWP1_ent.txt'),
                'HWP2': load_angles_from_file(self.path, 'Angoli_HWP2_ent.txt')
            })
            df_ent['label'] = 'ent'

        elif self.date in ['2024-11-08', '2025-01-09']:
            self.angles_labels = ['QWP1', 'QWP2', 'HWP1', 'HWP2']
            # 2024-11-08 uses different angles for the two walkers
            # but only one reference state for separable states
            df_sep = pd.DataFrame({
                'QWP1': load_angles_from_file(self.path, 'Angoli_QWP1_sep.txt'),
                'QWP2': load_angles_from_file(self.path, 'Angoli_QWP2_sep.txt'),
                'HWP1': load_angles_from_file(self.path, 'Angoli_HWP1_sep.txt'),
                'HWP2': load_angles_from_file(self.path, 'Angoli_HWP2_sep.txt')
            })
            # add a label column to the dataframe whose value alternates between 'sep1' and 'sep2'
            df_sep['label'] = 'sep'
            df_ent = pd.DataFrame({
                'QWP1': load_angles_from_file(self.path, 'Angoli_QWP1_ent.txt'),
                'QWP2': load_angles_from_file(self.path, 'Angoli_QWP2_ent.txt'),
                'HWP1': load_angles_from_file(self.path, 'Angoli_HWP1_ent.txt'),
                'HWP2': load_angles_from_file(self.path, 'Angoli_HWP2_ent.txt')
            })
            df_ent['label'] = 'ent'

        # reset the index and rename the columns: goal is to have a unique id per state
        # this is useful to connect each state with its counts
        # concatenate the two dataframes for separable and entangled states
        self.states_data = pd.concat([df_sep, df_ent], ignore_index=True)
        # add a column to the dataframe that contains the state number (unique id for each state)
        # in this case the id is just the index, but when data is loaded from raw files,
        # the id is specified in the files themselves. The goal of the state_id is to
        # tie the states with the counts data
        self.states_data['state_id'] = self.states_data.index

        return self


    def _load_states(self) -> Self:
        """Compute the input states for each label using the angles data.

        Assumes that self.states_data has been populated with the angles data already.
        """
        # use the angles to compute the INPUT states for each label (not the reference states, those are predefined above)
        # goal is to have a dataframe with angles and states for each label
        list_of_states = []
        for idx, row in self.states_data.iterrows():
            if self.date in ['2024-09-20', '2024-09-02']:
                unitary_walker1 = local_polarization_unitary(row['QWP'], row['HWP'])
                unitary_walker2 = local_polarization_unitary(row['QWP'], row['HWP'])
            elif self.date in ['2024-09-30', '2024-11-08', '2025-01-09']:
                unitary_walker1 = local_polarization_unitary(row['QWP1'], row['HWP1'])
                unitary_walker2 = local_polarization_unitary(row['QWP2'], row['HWP2'])
            full_unitary = np.kron(unitary_walker1, unitary_walker2)
            full_unitary = qutip.Qobj(full_unitary, dims=[[2, 2], [2, 2]])
            list_of_states.append((full_unitary * self.reference_states[row['label']]).full()[:, 0])

        self.states_data['state'] = list_of_states
        return self
    
    def _load_counts(self) -> Self:
        """Load the counts data from the txt files.
        
        This does NOT handle the raw data files, which are loaded separately (when needed).

        The counts data is stored in the self.counts numpy array.
        """
        # find the count files: these have the form ccSEP_rep_X.txt or ccENT_rep_X.txt
        counts_files = [file for file in os.listdir(self.path) if file.startswith('cc')]

        # define a function to extract the repetition number and the label from the filename
        def extract_info_from_filename(filename: str) -> tuple[str, int]:
            return filename[2:5].lower(), int(filename[10:11])
        # load data from the files. Each file contains a list of integers, written in scientific notation.
        # Each row contains 5 numbers. Each set of 5 consecutive rows (25 numbers) corresponds to a counts vector
        # Each of these is stored as an element of the resulting array.
        # We then store all the loaded counts in the self.counts array:
        # ---- the first index corresponds to the state_id,
        # ---- the second index corresponds to the repetition number
        # ---- the third index corresponds to the counts vector
        
        # initialise the counts numpy array of shape num_states x 25 x num_repetitions
        # the len(counts_files) // 2 is because we have two counts files per repetition (one for sep and one for ent); it's a pretty ugly solution ngl
        self.counts = np.zeros((self.states_data.shape[0], 25, len(counts_files) // 2))
        # each counts file contains the counts for a specific label (ent or sep) and repetition
        # For the 2024-09-30 dataset, the sep counts files contain the counts for both sep1 and sep2
        for file in counts_files:
            label, rep = extract_info_from_filename(file)
            with open(os.path.join(self.path, file), 'r') as f:
                lines = f.read().split()
                counts = np.asarray([int(float(line)) for line in lines])
                counts = counts.reshape(-1, 25)
                # extract the indices of self.states_data that correspond to the given label
                # note that for the 2024-09-30 dataset, for separables, the file just says 'sep', but states_data contains
                # both sep1 and sep2. This is why we just take the labels that contain `label` here
                indices = self.states_data[self.states_data['label'].str.startswith(label)]['state_id']
                # counts files are enumerated starting from 1 not 0, hence the -1
                self.counts[indices, :, rep - 1] = counts

        return self

    def _load_angles_data_from_raw_files(self) -> Self:
        """Load the angles data from the raw data files.
        
        This function is used to load the angles data from the raw data files, which are in the form of pickles.
        Each pickle contains a dataframe with the angles for one state.

        This is intended to be used as a REPLACEMENT for _load_angles_data, when the raw data is available.
        """
        repetition_folders = [folder for folder in os.listdir(self.path) if folder.startswith('Ripetizione')]
        if len(repetition_folders) == 0:
            raise ValueError('No repetition folders found in the given path. Are you sure you\'re using a dataset that comes with row data? (you\'re not)')
        # the various repetition folders actually all contain the same angles data, so we just use the first one
        state_data_sep = load_angles_from_raw_files(os.path.join(self.path, repetition_folders[0], "Separabili"))
        state_data_ent = load_angles_from_raw_files(os.path.join(self.path, repetition_folders[0], "Entangled"))

        if self.date == '2025-01-09':
            # for the 2025-01-09 data, the state_ids are done differently: separable and entangled states each have a set of ids going from 0 to 150
            # this means we can't use state_id as a unique identifier, which breaks stuff.
            # To fix it, we manually add to the entangled states 150 to their state_id
            state_data_ent['state_id'] += 150
        self.states_data = pd.concat([state_data_sep, state_data_ent], ignore_index=True)

        return self

    def _load_raw_counts(self):
        """Load the raw counts data from the pickle files.
        
        This takes the data from the folders having name "Ripetizione_X", and only those. Also take case that those are numbered from 0 onwards otherwise shit might break.
        The counts data is stored in the arrays self.counts, self.singles_1, and self.singles_2.
        Each of these is a numpy array of shape num_states x 25 x num_repetitions.
        The state_id is used to connect these counts data with the stuff in self.states_data.
        """
        # load row data, if it exists
        # This data comes from the folders with relative path "Ripetizione_X/Entangled|Separabili/Raw/name.pickle"
        # Each pickle in each folder is converted into a Series, and all of the series are put into the same dataframe
        repetition_folders = [folder for folder in os.listdir(self.path) if folder.startswith('Ripetizione')]
        if len(repetition_folders) == 0:
            raise ValueError('No repetition folders found in the given path. Are you sure you\'re using a dataset that comes with row data? (you\'re not)')

        self.counts = np.zeros((self.states_data.shape[0], 25, len(repetition_folders)))
        self.singles_1 = np.zeros((self.states_data.shape[0], 25, len(repetition_folders)))
        self.singles_2 = np.zeros((self.states_data.shape[0], 25, len(repetition_folders)))
        for idx, folder in enumerate(repetition_folders):
            raw_data_sep = load_raw_counts_from_files(os.path.join(self.path, folder, "Separabili", "Raw"))
            raw_data_ent = load_raw_counts_from_files(os.path.join(self.path, folder, "Entangled", "Raw"))
            # for the 2025-01-09 data, the state_ids are done differently: separable and entangled states each have a set of ids going from 0 to 150
            # this means we can't use state_id as a unique identifier, which breaks stuff.
            # To fix it, we manually add to the entangled states 150 to their state_id
            if self.date == '2025-01-09':
                raw_data_ent['state_id'] += 150
            raw_data = pd.concat([raw_data_sep, raw_data_ent], ignore_index=True)
            self.counts[raw_data['state_id'].values, :, idx] = np.stack(list(raw_data['doubles'].values))
            self.singles_1[raw_data['state_id'].values, :, idx] = np.stack(list(raw_data['singles_1'].values))
            self.singles_2[raw_data['state_id'].values, :, idx] = np.stack(list(raw_data['singles_2'].values))
            if not self.stfu:
                display(Markdown(f'***Loaded raw data from folder*** *`{folder}`*'))
            
        return self


    def merge_repetitions(self, mean_or_sum='mean'):
        """Merge the repetitions of the data in counts, singles_1, and singles_2.
        
        This function is used to sum or average the counts for each state across all repetitions.

        Parameters
        ----------
        mean_or_sum : str, optional
            Whether to compute the mean or the sum of the counts across all repetitions.
            Default is 'mean'. Other possible values are 'sum'.
            Summing rather than averaging is generally problematic if you use different
            numbers of repetitions in training and test. Try it if you don't believe me.
        """
        if len(self.counts.shape) == 2:
            # in this case we don't have repetitions, so don't do anything
            return self
        if mean_or_sum == 'mean':
            self.counts = np.mean(self.counts, axis=2)
            if hasattr(self, 'singles_1') and self.singles_1 is not None:
                self.singles_1 = np.mean(self.singles_1, axis=2)
                self.singles_2 = np.mean(self.singles_2, axis=2)
        elif mean_or_sum == 'sum':
            self.counts = np.sum(self.counts, axis=2)
            if hasattr(self, 'singles_1') and self.singles_1 is not None:
                self.singles_1 = np.sum(self.singles_1, axis=2)
                self.singles_2 = np.sum(self.singles_2, axis=2)

        return self
    
    def save_all_data_to_file(self):
        # save the all_data dataframe as a pickle
        # when existing in a folder, the data will be loaded from there instead of recomputed from the raw datafiles
        all_data = (self.states_data, self.counts)
        if self.singles_1 is not None:
            all_data += (self.singles_1, self.singles_2)
        # dump the data to a pickle file
        with open(os.path.join(self.path, 'all_data.pickle'), 'wb') as f:
            pickle.dump(all_data, f)
    
    def marginalize_singles_columns(self):
        """Marginalize the singles columns for each walker.
        
        More specifically, singles_1 has each row summed, and singles_2 same with cols.
        This is because by default singles are stored as vectors of length 25. We thus reshape them into 5x5
        matrices and sum over the rows or columns.
        """
        # if there's 3 dimensions then we still have repetitions
        if len(self.singles_1.shape) == 3:
            num_reps = self.singles_1.shape[2]
            self.singles_1 = np.einsum('ijkl->ijl', self.singles_1.reshape(-1, 5, 5, num_reps))
            self.singles_2 = np.einsum('ijkl->ikl', self.singles_2.reshape(-1, 5, 5, num_reps))
        # if we've already merged the repetitions, then we only have 3 dimensions
        else:
            self.singles_1 = self.singles_1.reshape(-1, 5, 5).sum(axis=2)
            self.singles_2 = self.singles_2.reshape(-1, 5, 5).sum(axis=1)

        return self

    def filter_rows(self, rows_specs) -> Self:
        """Filter the rows of the states_data dataframe.
        
        Note that this filters the rows of states_data *in place*.
        You might want to use `filter_all_data_rows` instead if you only need to extract different
        subsets of the data without modifying the original dataframe.

        NOTE: This doesn't touch counts or singles arrays. Which is fine as long as you don't just 
              try to use them without using the suitable state_id to connect them to the filtered states_data.

        Parameters
        ----------
        rows_specs : str or pd.DataFrame
            If a string, it's used to decide among predefined rows.
            If a DataFrame, it's used as a mask for the states_data dataframe.
        """
        if isinstance(rows_specs, str):
            if rows_specs == 'sep1':
                self.states_data = self.states_data[self.states_data['label'] == 'sep1']
                self.labels = ['sep1']
            elif rows_specs == 'sep2':
                self.states_data = self.states_data[self.states_data['label'] == 'sep2']
                self.labels = ['sep2']
            elif rows_specs == 'allsep':
                self.states_data = self.states_data[self.states_data['label'].str.contains('sep')]
                self.labels = ['sep1', 'sep2']
            elif rows_specs == 'ent':
                self.states_data = self.states_data[self.states_data['label'] == 'ent']
                self.labels = ['ent']
        else:
            # if not a string, it's assumed to be a dataframe with bool values
            # to be used as a mask for the states_data dataframe
            self.states_data = self.states_data[rows_specs]
        
        return self
    
    def compute_target_expvals(self, target_observables):
        """Compute expectation values for each state in states_data.
        
        Compute the expectation values of the specified observables, for each input state.
        The states are taken from the 'state' column of the states_data dataframe.

        Parameters
        ----------
        target_observables : list of qutip.Qobj
            The observables for which to compute the expectation values.
            These are assumed to be qutip objects because I'm lazy.
        """
        expvals = []
        for _, row in self.states_data.iterrows():
            state = qutip.Qobj(row['state'], dims=[[2, 2], [1, 1]])
            expvals.append([qutip.expect(obs, state) for obs in target_observables])
        # add the expectation values to the states_data dataframe
        self.states_data['expvals'] = expvals

        return self

    def get_training_dataset(self,
        which_states: str | pd.Series | np.ndarray = 'all',
        merge_reps: str | bool = False,
        target_observables=None,
        which_counts='doubles'
    ) -> dict:
        """
        Get the data needed for training a model.
        
        Parameters
        ----------
        which_states : str or pd.Series or np.ndarray of bool, optional
            Which states to include in the dataset. Can be 'all', 'sep', 'ent', 'sep1', 'sep2',
            or a boolean mask (pd.Series or np.ndarray). Default is 'all'.
        merge_reps : str or bool, optional
            Whether to merge the repetitions of the data. Can be 'mean', 'sum', or False.
            Default is 'mean'.
        target_observables : list of qutip.Qobj, optional
            Observables for which to compute expectation values (if not already computed).
        which_counts : str, optional
            Which counts to use. Can be 'doubles', 'singles_1', or 'singles_2'.
            Default is 'doubles'.

        Returns
        -------
        dict
            A dictionary with the data needed for training. Keys:
            - 'counts': numpy array of shape num_outcomes x num_states
            - 'expvals': numpy array of shape num_observables x num_states
        """
        if merge_reps == 'sum':
            self.merge_repetitions('sum')
        elif merge_reps == 'mean':
            self.merge_repetitions('mean')

        if target_observables is not None:
            self.compute_target_expvals(target_observables)

        if which_counts == 'doubles':
            counts = self.counts
        elif which_counts == 'singles_1':
            counts = self.singles_1
        elif which_counts == 'singles_2':
            counts = self.singles_2

        states_data = self.states_data[['state_id', 'label', 'expvals']].copy(deep=True)

        if isinstance(which_states, str):
            if which_states == 'all':
                pass
            elif which_states == 'sep' or which_states == 'allsep':
                states_data = states_data.loc[states_data['label'].str.contains('sep')]
            elif which_states == 'sep1':
                states_data = states_data.loc[states_data['label'] == 'sep1']
            elif which_states == 'sep2':
                states_data = states_data.loc[states_data['label'] == 'sep2']
            elif which_states == 'ent':
                states_data = states_data.loc[states_data['label'] == 'ent']
        else:
            # otherwise assume it's a boolean mask to directly select which states to include
            states_data = states_data.loc[which_states]

        states_data.sort_values('state_id', inplace=True)
        good_indices = np.asarray(states_data['state_id'].values)

        dict_data = {
            'counts': counts[good_indices].T,
            'expvals': np.stack(list(states_data['expvals'].values)).T
        }
        return dict_data
