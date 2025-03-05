import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import importlib
import os
import sys

import qutip
from pprint import pprint
from IPython.display import display, HTML, Markdown
import seaborn as sns
sns.set_theme()
import plotly
import plotly.express as px
import plotly.subplots as sp
# Tomas Mazak's workaround
plotly.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))

# list of all 16 two-qubit pauli operators
one_qubit_paulis = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
two_qubit_paulis = [qutip.tensor(p1, p2) for p1 in one_qubit_paulis for p2 in one_qubit_paulis]
two_qubit_paulis_labels = ['I_1I_2', 'I_1X_2', 'I_1Y_2', 'I_1Z_2', 'X_1I_2', 'X_1X_2', 'X_1Y_2', 'X_1Z_2', 'Y_1I_2', 'Y_1X_2', 'Y_1Y_2', 'Y_1Z_2', 'Z_1I_2', 'Z_1X_2', 'Z_1Y_2', 'Z_1Z_2']
# list with the projections onto the four bell states
bell_states = [qutip.bell_state('00'), qutip.bell_state('01'), qutip.bell_state('10'), qutip.bell_state('11')]
# witness operators for the four bell states
bell_witnesses = [qutip.qeye([2, 2]) / 2 - bell_state * bell_state.dag() for bell_state in bell_states]

# add absolute current path to path
sys.path.append(os.path.abspath('.'))

import ExperimentalDataset
importlib.reload(ExperimentalDataset)
from ExperimentalDataset import ExperimentalDataset

import QELM
importlib.reload(QELM)

from utils import train_and_test_QELM_on_doubles, plot_paulis_scatter
from utils import plot_witnesses_scatter, plot_singular_values_counts, check_singular_values_states

dir_experimental_data_0902 = os.path.join('..', 'experimental data', 'dati 2024-09-02')
dir_experimental_data_0920 = os.path.join('..', 'experimental data', 'dati 2024-09-20')
dir_experimental_data_0930 = os.path.join('..', 'experimental data', 'dati 2024-09-30')
dir_experimental_data_1108 = os.path.join('..', 'experimental data', 'dati 2024-11-08')
dir_experimental_data_0109 = os.path.join('..', 'experimental data', 'dati 2025-01-09')
dir_experimental_data_0121 = os.path.join('..', 'experimental data', 'dati 2025-01-21')
dir_experimental_data_1212 = os.path.join('..', 'experimental data', 'dati 2024-12-12')
