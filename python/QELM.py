import os
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip
from IPython.display import display, Markdown
from typing import Optional


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

    def _train_standard(self, truncate_pseudo_inverse=False, rcond=None):
        """
        Train using the standard method.

        Parameters:
        -----------
        truncate_pseudo_inverse : bool or int
            Whether to truncate the pseudo-inverse calculation at a finite value. Default is False.
            This means to only take a finite number of singular values before doing the pseudoinverse.
        rcond : array_like or float, optional
            This is passed over to numpy.linalg.pinv
        """
        if self.train_dict is None:
            raise ValueError('Train data not provided.')

        self.W = np.dot(
            self.train_dict['expvals'],
            np.linalg.pinv(self.train_dict['counts'])
        )

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