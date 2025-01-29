# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
import plotly
import plotly.express as px
import plotly.subplots as sp

import QELM
from ExperimentalDataset import ExperimentalDataset

# labels for the two-qubit pauli operators
two_qubit_paulis_labels = ['I_1I_2', 'I_1X_2', 'I_1Y_2', 'I_1Z_2', 'X_1I_2', 'X_1X_2', 'X_1Y_2', 'X_1Z_2', 'Y_1I_2', 'Y_1X_2', 'Y_1Y_2', 'Y_1Z_2', 'Z_1I_2', 'Z_1X_2', 'Z_1Y_2', 'Z_1Z_2']

def train_and_test_QELM_on_doubles(data_dir, which_states_train, which_states_test,
                                   target_observables, mean_or_sum='mean',
                                   which_reps_train='all', which_reps_test='all',
                                   stfu=True, train_options={}):
    # extract data from experimental files
    experimentalDataset = ExperimentalDataset(data_dir, stfu=stfu)
    # merge repetitions (either summing or taking the average over the repetitions)
    # experimentalDataset.merge_repetitions(mean_or_sum=mean_or_sum)
    train_dict = experimentalDataset.get_training_dataset(
        which_states=which_states_train, which_counts='doubles',
        target_observables=target_observables
    )    
    test_dict = experimentalDataset.get_training_dataset(
        which_states=which_states_test, which_counts='doubles',
        target_observables=target_observables
    )
    # if which_reps_train is an integer, convert it to a list with that integer (and same for test)
    # this is to ensure the slicing afterwards works correctly even if the function is called with `0` rather than `[0]`
    if isinstance(which_reps_train, int):
        which_reps_train = [which_reps_train]
    if isinstance(which_reps_test, int):
        which_reps_test = [which_reps_test]
    # use only the repetitions specified in which_reps_train and which_reps_test
    if which_reps_train != 'all':
        try:
            train_dict['counts'] = train_dict['counts'][which_reps_train]
        except IndexError as e:
            raise ValueError(f"Invalid which_reps_train indices: {which_reps_train}") from e
    if which_reps_test != 'all':
        try:
            test_dict['counts'] = test_dict['counts'][which_reps_test]
        except IndexError as e:
            raise ValueError(f"Invalid which_reps_test indices: {which_reps_test}") from e
    # average or sum over the requested repetitions
    if mean_or_sum == 'mean':
        train_dict['counts'] = train_dict['counts'].mean(axis=0)
        test_dict['counts'] = test_dict['counts'].mean(axis=0)
    elif mean_or_sum == 'sum':
        train_dict['counts'] = train_dict['counts'].sum(axis=0)
        test_dict['counts'] = test_dict['counts'].sum(axis=0)
    # create and train the QELM
    qelm = QELM.QELM(train_dict=train_dict, test_dict=test_dict, train_options=train_options)
    qelm.compute_MSE(display_results=False)
    return qelm

def plot_witnesses_scatter(trained_qelm, plotly=False):
    if not plotly:
        plot_witnesses_scatter_noplotly(trained_qelm)
        return
    raise ValueError('Plotly plotting not implemented yet. Just use plotly=False')


def plot_witnesses_scatter_noplotly(trained_qelm):
    """Plots the scatters for the 4 witnesses."""
    witnesses_labels = [r'$\Phi^+$', r'$\Phi^-$', r'$\Psi^+$', r'$\Psi^-$']
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for i in range(4):
        if trained_qelm.train_dict is not None:
            axs[i // 2, i % 2].scatter(
                trained_qelm.train_dict['expvals'][i],
                trained_qelm.train_predictions[i],
                color='red', label='train', s=12
            )
        if trained_qelm.test_dict is not None:
            axs[i // 2, i % 2].scatter(
                trained_qelm.test_dict['expvals'][i],
                trained_qelm.test_predictions[i],
                color='blue', label='test', s=12
            )
        axs[i // 2, i % 2].plot([-1.1, 1.1], [-1.1, 1.1], 'k--')
        axs[i // 2, i % 2].axhline(0, color='black', linewidth=0.5)
        axs[i // 2, i % 2].axvline(0, color='black', linewidth=0.5)
        axs[i // 2, i % 2].text(0.1, 0.9, witnesses_labels[i], transform=axs[i // 2, i % 2].transAxes, 
                                horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        axs[i // 2, i % 2].set_xlim([-0.7, 0.7])
        axs[i // 2, i % 2].set_ylim([-0.7, 0.7])
        if i // 2 == 1:
            axs[i // 2, i % 2].set_xlabel('true')
        if i % 2 == 0:
            axs[i // 2, i % 2].set_ylabel('predicted')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_paulis_scatter(trained_qelm, plotly=True):

    if not plotly:
        plot_paulis_scatter_noplotly(trained_qelm)
        return

    labels = two_qubit_paulis_labels
    fig = sp.make_subplots(rows=4, cols=4,
                           subplot_titles=[f'${labels[i]}$' for i in range(16)],
                           shared_xaxes=True, shared_yaxes=True,
                           vertical_spacing=0.05, horizontal_spacing=0.05)
    
    for i in range(16):
        if trained_qelm.train_dict is not None:
            train_scatter = px.scatter(
                x=trained_qelm.train_dict['expvals'][i],
                y=trained_qelm.train_predictions[i],
                labels={'x': 'true', 'y': 'predicted'},
                color_discrete_sequence=['red']
            ).update_traces(marker=dict(size=4)).data[0]
            train_scatter.name = 'train'
            train_scatter.showlegend = i == 0
            fig.add_trace(train_scatter, row=(i // 4) + 1, col=(i % 4) + 1)

        if trained_qelm.test_dict is not None:
            test_scatter = px.scatter(
                x=trained_qelm.test_dict['expvals'][i],
                y=trained_qelm.test_predictions[i],
                labels={'x': 'true', 'y': 'predicted'},
                color_discrete_sequence=['green']
            ).update_traces(marker=dict(size=4)).data[0]
            test_scatter.name = 'test'
            test_scatter.showlegend = i == 0
            fig.add_trace(test_scatter, row=(i // 4) + 1, col=(i % 4) + 1)
        # also plot a diagonal dashed line with slope 1
        fig.add_shape(
            type='line',
            x0=-1, y0=-1, x1=1, y1=1,
            line=dict(color='black', width=1, dash='dash'),
            row=(i // 4) + 1, col=(i % 4) + 1
        )
    fig.update_layout(
        height=600,
        width=600
    )
    fig.show()

def plot_paulis_scatter_noplotly(trained_qelm):
    labels = two_qubit_paulis_labels
    fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(16):
        if trained_qelm.train_dict is not None:
            axs[i // 4, i % 4].scatter(
                trained_qelm.train_dict['expvals'][i],
                trained_qelm.train_predictions[i],
                color='red', label='train', s=12
            )
        if trained_qelm.test_dict is not None:
            axs[i // 4, i % 4].scatter(
                trained_qelm.test_dict['expvals'][i],
                trained_qelm.test_predictions[i],
                color='blue', label='test', s=12
            )
        axs[i // 4, i % 4].plot([-1.1, 1.1], [-1.1, 1.1], 'k--')
        axs[i // 4, i % 4].text(0.5, 0.9, f'${labels[i]}$', transform=axs[i // 4, i % 4].transAxes, 
                                horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        axs[i // 4, i % 4].set_xlim([-1.1, 1.1])
        axs[i // 4, i % 4].set_ylim([-1.1, 1.1])
        if i // 4 == 3:
            axs[i // 4, i % 4].set_xlabel('true')
        if i % 4 == 0:
            axs[i // 4, i % 4].set_ylabel('predicted')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_singular_values_counts(data_dir, per_repetition=False):
    """Plots the singular values of the counts matrix for each repetition.
    
    If per_repetition is True, the singular values are plotted for each repetition separately.
    Otherwise the repetitions are averaged before computing the singular values.
    """
    experimentalDataset = ExperimentalDataset(data_dir)
    if per_repetition:
        counts = experimentalDataset.counts
        for i in range(counts.shape[2]):
            singular_values = np.linalg.svd(counts[:, :, i], compute_uv=False)
            plt.plot(singular_values, label=f'rep {i}', marker='o')
        plt.xlabel('Singular value index')
        plt.ylabel('Singular value')
        plt.legend()
        plt.show()
    else:
        experimentalDataset.merge_repetitions(mean_or_sum='mean')
        counts = experimentalDataset.counts
        singular_values = np.linalg.svd(counts, compute_uv=False)
        plt.plot(singular_values, marker='o')
        plt.xlabel('Singular value index')
        plt.ylabel('Singular value')
        plt.show()


def check_singular_values_states(data_dir, which_labels):
    """Prints the singular values of the vectorized states matrices for the specified states."""
    experimentalDataset = ExperimentalDataset(data_dir)
    states = experimentalDataset.states_data
    for label in which_labels:
        if label != 'all':
            states_with_label = states.loc[states['label'].str.contains(label)]['state'].values
        else:
            states_with_label = states['state'].values
        states_with_label = np.array(list(states_with_label))  # pd.values returns an array of dtype object, while we want an array of dtype complex
        vectorized_states = QELM.kets_to_vectorized_states_matrix(states_with_label, basis='paulis')
        singular_values = np.linalg.svd(vectorized_states, compute_uv=False)
        print(f'Singular values for the {states_with_label.shape[0]} states with label `{label}`:\n{singular_values}')