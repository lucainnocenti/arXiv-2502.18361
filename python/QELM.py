import os
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip
from IPython.display import display, Markdown
from typing import Optional


def truncate_svd(matrix, singular_values_kept):
    """
    Truncate the singular value decomposition (SVD) of a matrix.

    This function takes a matrix, computes its singular value decomposition (SVD),
    and truncates all singular values beyond a specified number.

    Parameters:
    matrix (ndarray): The input matrix to decompose.
    singular_values_kept (int): The number of singular values to keep.

    Returns:
    ndarray: The matrix reconstructed from the truncated SVD.
    """
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    S_truncated = np.zeros_like(S)
    S_truncated[:singular_values_kept] = S[:singular_values_kept]
    return U @ np.diag(S_truncated) @ V


def vectorize_density_matrix(rho, basis='paulis'):
    """
    Vectorize a density matrix.

    This function takes a density matrix and vectorizes it in some specified basis.

    Parameters:
    rho (ndarray): The density matrix to vectorize.

    Returns:
    ndarray: The vectorized density matrix.
    """
    # if basis is a string then we assume it's a standard basis
    if basis == 'flatten':
        return rho.flatten()
    # if it's a string and it equals 'paulis' then we vectorize in the Pauli basis
    elif basis == 'paulis':
        single_qubit_ops = [np.eye(2), qutip.sigmax().full(), qutip.sigmay().full(), qutip.sigmaz().full()]
        # single_qubit_ops = [op / np.sqrt(2) for op in single_qubit_ops]  # this normalization makes the basis self-dual
        if rho.shape[0] == 2:
            # it's a single-qubit density matrix
            op_basis = single_qubit_ops
        elif rho.shape[0] == 4:
            # it's a two-qubit density matrix, so take all tensor products of single-qubit operators
            op_basis = [np.kron(op1, op2) for op1 in single_qubit_ops for op2 in single_qubit_ops]
        else:
            raise ValueError('Only 1 and 2 qubits for now.')
        return np.array([np.trace(rho @ op) for op in op_basis])


def devectorize_density_matrix(rho_vec, basis='paulis'):
    """
    Devectorize a density matrix.

    This function takes a vectorized density matrix and devectorizes it.
    It's intended to be an inverse of vectorize_density_matrix.

    Parameters:
    rho_vec (ndarray): The vectorized density matrix.

    Returns:
    ndarray: The density matrix.
    """
    d = int(np.sqrt(len(rho_vec)))  # dimension of the density matrix
    if basis == 'flatten':
        return rho_vec.reshape((d, d))
    elif basis == 'paulis':
        single_qubit_ops = [np.eye(2), qutip.sigmax().full(), qutip.sigmay().full(), qutip.sigmaz().full()]
        single_qubit_ops = [op / 2 for op in single_qubit_ops]  # to ensure proper inversion
        if d == 2:
            op_basis = single_qubit_ops
        elif d == 4:
            op_basis = [np.kron(op1, op2) for op1 in single_qubit_ops for op2 in single_qubit_ops]
        else:
            raise ValueError('Only 1 and 2 qubits for now.')
        return sum([rho_vec[i] * op for i, op in enumerate(op_basis)])
    return rho_vec.reshape((d, d))


def ket_to_dm(ket):
    """Works with numpy arrays."""
    # ensure that ket is a column vector
    ket = np.asarray(ket).reshape(-1, 1)
    return ket * ket.conj().T


def kets_to_vectorized_states_matrix(list_of_kets, basis='paulis'):
    """
    Convert a list of kets to a vectorized states matrix.

    This function takes a list of kets and converts them to a matrix of vectorized states.
    The result is a matrix of dimensions (dim^2, num_states), where dim is the dimension of the density matrix.

    Parameters:
    list_of_kets (list): The list of kets to convert.
    basis (str): The basis in which to vectorize the states.

    Returns:
    ndarray: The matrix of vectorized states.
    """
    return np.array([vectorize_density_matrix(ket_to_dm(ket), basis=basis) for ket in list_of_kets]).T


class QELM:
    """
    Quantum Extreme Learning Machine (QELM) class for training and testing.
    """
    def __init__(self, train_dict: Optional[dict] = None, test_dict: Optional[dict] = None,
                 method: str = 'standard', train_options: Optional[dict] = None,
                 W: Optional[np.ndarray] = None):
        """
        Initialize the QELM class.

        Parameters:
        -----------
        train_dict : dict
            probabilities and expectation values for training.
        test_dict : dict
            probabilities and expectation values for testing
        method (str): Method to be used for training. Default is 'standard'.
        """
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.method = method
        self.train_predictions: np.ndarray
        self.test_predictions: np.ndarray

        if train_options is None:
            train_options = {}

        # if W is given we don't need to train
        if W is not None:
            self.W = W
        else:
            self._train_standard(**train_options)

    def _train_standard(self, truncate_singular_values=False, rcond=None):
        """
        Train using the standard method.

        Parameters:
        -----------
        truncate_singular_values : bool or int
            Whether to truncate the pseudo-inverse calculation at a finite value. Default is False.
            This means to only take a finite number of singular values before doing the pseudoinverse.
        rcond : array_like or float, optional
            This is passed over to numpy.linalg.pinv
        """
        if self.train_dict is None:
            raise ValueError('Train data not provided.')

        counts = self.train_dict['counts']
        if truncate_singular_values is not False:
            # assume it's an integer
            counts = truncate_svd(counts, truncate_singular_values)

        self.W = np.dot(
            self.train_dict['expvals'],
            np.linalg.pinv(counts)
        )
    
    def compute_state_shadow(self, truncate_singular_values=False):
        """
        Compute the state shadow of the QELM.
        
        Meaning M_rho^T @ counts. This gives a matrix which can then later provide the estimator for any target
        observable.
        Effectively, it's a matrix that applied to any test estimated probability returns an estimated full tomographic
        reconstruction of the measured state. I.e. it's the classical shadow estimator.
        """
        if self.train_dict is None:
            raise ValueError('Train data not provided.')

        counts = self.train_dict['counts']        
        if len(counts.shape) == 3:
            # 3 dimensions implies that we have repetitions, which ain't nice. Throw an error.
            raise ValueError('The counts array has 3 dimensions, which implies repetitions. Please cut down to one repetition.')

        if truncate_singular_values is not False:
            # assume it's an integer
            counts = truncate_svd(counts, truncate_singular_values)
        # self.shadow_estimator = np.dot()
        # check whether self.train_dict has a 'state' key, throw an error if it doesn't
        if 'states' not in self.train_dict:
            raise ValueError('No state provided in train_dict. Did you use `save_states=True` when creating the dataset?')
        # convert the states (which are usually stored as kets) to density matrices, and then vectorize them
        states_matrix = kets_to_vectorized_states_matrix(self.train_dict['states'], basis='paulis')
        # states_matrix = np.array([vectorize_density_matrix(ket_to_dm(state), basis='paulis') for state in self.train_dict['states']])
        # # states_matrix has now size num_states x dim^2, so we transpose it to adhere to our nice analytical conventions
        # states_matrix = states_matrix.T
        # NOW compute the state shadow
        self.state_shadow = np.dot(states_matrix, np.linalg.pinv(counts))
        return self


    def predict(self, probabilities):
        """
        Predict the expectation values for given probabilities.

        Parameters:
        -----------
        probabilities : np.ndarray
            Probabilities for which to predict the expectation values.

        Returns:
        --------
        np.ndarray
            Predicted expectation values.
        """
        return np.dot(self.W, probabilities)

    def compute_MSE(self, train=True, test=True, display_results=True):
        """
        Test the trained model on train and test data computing the mean squared error.
        """
        # check test data has been provided
        if self.test_dict is None:
            raise ValueError('Test data not provided.')

        
        if train:
            if self.train_dict is None:
                raise ValueError('Train data not provided.')
            self.train_predictions = self.predict(self.train_dict['counts'])
            self.train_MSE = np.mean((self.train_predictions - self.train_dict['expvals']) ** 2, axis=1)
        if test:
            self.test_predictions = self.predict(self.test_dict['counts'])
            self.test_MSE = np.mean((self.test_predictions - self.test_dict['expvals']) ** 2, axis=1)

        if display_results:
            display(Markdown(f"***Train MSE***: {self.train_MSE}"))
            display(Markdown(f"***Test MSE***: {self.test_MSE}"))
        
        return self