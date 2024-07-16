# importing required libraries
import numpy as np
import pandas as pd
import h5py
import pints
import os
import sys
import argparse
import pints.toy
import pints.plot

import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pints.toy import ToyModel

class SIRModel(pints.ForwardModel, ToyModel):
    r"""
    The SIR model of infectious disease models the number of susceptible (S),
    infected (I), and recovered (R) people in a population [1]_, [2]_.

    The particular model given here is analysed in [3],_ and is described by
    the following three-state ODE:

    .. math::
        \dot{S} = -\gamma S I

        \dot{I} = \gamma S I - v I

        \dot{R} = v I

    Where the parameters are ``gamma`` (infection rate), and ``v``, recovery
    rate. In addition, we assume the initial value of S, ``S0``, is unknwon,
    leading to a three parameter model ``(gamma, v, S0)``.

    The number of infected people and recovered people are observable, making
    this a 2-output system. S can be thought of as an unknown number of
    susceptible people within a larger population.

    The model does not account for births and deaths, which are assumed to
    happen much slower than the spread of the (non-lethal) disease.

    Real data is included via :meth:`suggested_values`, which was taken from
    [3]_, [4]_, [5]_.

    Extends :class:`pints.ForwardModel`, `pints.toy.ToyModel`.

    Parameters
    ----------
    y0
        The system's initial state, must have 3 entries all >=0.

    References
    ----------
    .. [1] A Contribution to the Mathematical Theory of Epidemics. Kermack,
           McKendrick (1927) Proceedings of the Royal Society A.
           https://doi.org/10.1098/rspa.1927.0118

    .. [2] https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

    .. [3] Approximate Bayesian computation scheme for parameter inference and
           model selection in dynamical systems. Toni, Welch, Strelkowa, Ipsen,
           Stumpf (2009) J. R. Soc. Interface.
           https://doi.org/10.1098/rsif.2008.0172

    .. [4] A mathematical model of common-cold epidemics on Tristan da Cunha.
           Hammond, Tyrrell (1971) Epidemiology & Infection.
           https://doi.org/10.1017/S0022172400021677

    .. [5] Common colds on Tristan da Cunha. Shybli, Gooch, Lewis, Tyrell
           (1971) Epidemiology & Infection.
           https://doi.org/10.1017/S0022172400021483
    """

    def __init__(self,N,y0=None):
        super(SIRModel, self).__init__()

        # Check initial values
        if y0 is None:
            # Toni et al.:
            self._y0 = np.array([38, 1, 0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError('Initial value must have size 3.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')
        self.N=N



    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 3





    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 3



    def _rhs(self, y, t, gamma, v):
        """
        Calculates the model RHS.
        """
        dS = (-gamma * y[0] * y[1])/self.N
        dI = (gamma * y[0] * y[1])/self.N - v * y[1]
        dR = v * y[1]
        return np.array([dS, dI, dR])



    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        gamma, v, S0 = parameters
        y0 = np.array(self._y0, copy=True)
        y0[0] = S0
        y = odeint(self._rhs, y0, times, (gamma, v))
        return y[:, :]





    def suggested_parameters(self):
        """
        Returns a suggested set of parameters for this toy model.
        """
        # Guesses based on Toni et al.:
        return [0.026, 0.285, 38]





    def suggested_times(self):
        """
        Returns a suggested set of simulation times for this toy model.
        """
        # Toni et al.:
        return np.arange(1, 22)





    def suggested_values(self):
        """
        Returns the data from a common-cold outbreak on the remote island of
        Tristan da Cunha, as given in [3]_, [4]_, [5]_.
        """
        # Toni et al.
        return np.array([
            [1, 0],     # day 1
            [1, 0],
            [3, 0],
            [7, 0],
            [6, 5],     # day 5
            [10, 7],
            [13, 8],
            [13, 13],
            [14, 13],
            [14, 16],    # day 10
            [17, 17],
            [10, 24],
            [6, 30],
            [6, 31],
            [4, 33],    # day 15
            [3, 34],
            [1, 36],
            [1, 36],
            [1, 36],
            [1, 36],    # day 20
            [0, 37],    # day 21
        ])

# simulator function that takes the combination and the number of samples per combination and the suggested times
def simulator(combination,P,times=None):
    # If times is not provided, use the suggested times
    if times is None:
        times = pints.toy.SIRModel().suggested_times()
    
    
    y = combination[:3]
    # c is the 4,5, and first element of combination
    c = [combination[3], combination[4], combination[0]]
    # Create the model
    model = SIRModel(y0=y,N=P)
    # Simulate the model
    data = model.simulate(c, times)
    
    # Add noise to the data
    #noise = np.random.normal(0, 0.1, data.shape)
    #data += noise
    # Return the data
    return data


# combination & number of samples
def generate_data(n, num_samples):
    combination = []
    # combination count
    count = 0
    # total population
    N = n

    while count < num_samples:
        I0 = np.random.randint(1000,N/2)
        R0 = np.random.randint(N-I0)
        S0 = N - I0 - R0
        gamma = np.random.uniform(0.1, 0.5)
        v = np.random.uniform(0.01, 0.2)
        c = [S0, I0, R0, gamma, v]
        # check if the combination is not in combination list
        if c not in combination:
            combination.append(c)
            count += 1
        else:
            continue
    return combination


if __name__ == '__main__':

    wd = os.getcwd()
    print(wd)
    
    parse = argparse.ArgumentParser(description="Perform basic mathematical operations.")
    parse.add_argument('--output_path', type=str, required=True, help='Path to save the output')
    parse.add_argument('--train_path', type=str, required=True, help='Path to save the training dataset')
    parse.add_argument('--valid_path', type=str, required=True, help='Path to save the validation dataset')
    parse.add_argument('--test_path', type=str, required=True, help='Path to save the test dataset')
    parse.add_argument('--N', type=int, required=True, help='Number of combinations')
    parse.add_argument('--population', type=int, required=True, help='population size')
    args = parse.parse_args()

    
    # paths to save the datasets
    output_path = args.output_path
    train_data_path = os.path.join(wd,output_path,args.train_path)
    valid_data_path = os.path.join(wd,output_path,args.valid_path)
    test_data_path = os.path.join(wd,output_path,args.test_path)
    # number of population
    p = args.population
    # number of samples per combination
    num_samples = args.N

    train_path = train_data_path
    valid_path = valid_data_path
    test_path = test_data_path
    # number of samples per combination
    times = np.linspace(0, 180, 1025)
    # generate the data
    combination = generate_data(p, num_samples)

    # split combination into training, validation and test sets (80%, 10%, 10%) of the combination
    train_size = int(0.8 * len(combination))
    valid_size = int(0.1 * len(combination))
    test_size = len(combination) - train_size - valid_size

    train_comb = combination[:train_size]
    valid_comb = combination[train_size:train_size + valid_size]
    test_comb = combination[-test_size:]

    # make the directories
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(valid_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    # create the training dataset
    with h5py.File(train_path, 'w') as f:
        for i, data in enumerate(train_comb):
            values = simulator(data,p,times)
            f.create_dataset(str(i), data=values)
    
    # create the validation dataset
    with h5py.File(valid_path, 'w') as f:
        for i, data in enumerate(valid_comb):
            values = simulator(data,p,times)
            f.create_dataset(str(i), data=values)

    # create the test dataset
    with h5py.File(test_path, 'w') as f:
        for i, data in enumerate(test_comb):
            values = simulator(data,p,times)
            f.create_dataset(str(i), data=values) 
