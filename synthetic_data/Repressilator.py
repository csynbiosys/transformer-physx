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


# Repressilator model
class RepressilatorModel(pints.ForwardModel, ToyModel):
    """
    The "Repressilator" model describes oscillations in a network of proteins
    that suppress their own creation [1]_, [2]_.

    The formulation used here is taken from [3]_ and analysed in [4]_. It has
    three protein states (:math:`p_i`), each encoded by mRNA (:math:`m_i`).
    Once expressed, they suppress each other:

    .. math::
        \\dot{m_0} = -m_0 + \\frac{\\alpha}{1 + p_2^n} + \\alpha_0

        \\dot{m_1} = -m_1 + \\frac{\\alpha}{1 + p_0^n} + \\alpha_0

        \\dot{m_2} = -m_2 + \\frac{\\alpha}{1 + p_1^n} + \\alpha_0

        \\dot{p_0} = -\\beta (p_0 - m_0)

        \\dot{p_1} = -\\beta (p_1 - m_1)

        \\dot{p_2} = -\\beta (p_2 - m_2)

    With parameters ``alpha_0``, ``alpha``, ``beta``, and ``n``.

    Only the mRNA states are visible as output.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    y0
        The system's initial state, must have 6 entries all >=0.

    References
    ----------
    .. [1] A Synthetic Oscillatory Network of Transcriptional Regulators.
          Elowitz, Leibler (2000) Nature.
          https://doi.org/10.1038/35002125

    .. [2] https://en.wikipedia.org/wiki/Repressilator

    .. [3] Dynamic models in biology. Ellner, Guckenheimer (2006) Princeton
           University Press

    .. [4] Approximate Bayesian computation scheme for parameter inference and
           model selection in dynamical systems. Toni, Welch, Strelkowa, Ipsen,
           Stumpf (2009) J. R. Soc. Interface.
           https://doi.org/10.1098/rsif.2008.0172
    """

    def __init__(self, y0=None):
        super(RepressilatorModel, self).__init__()

        # Check initial values
        if y0 is None:
            # Toni et al.:
            self._y0 = np.array([0, 0, 0, 2, 1, 3])
            # Figure 42 in book
            #self._y0 = np.array([0.2, 0.1, 0.3, 0.1, 0.4, 0.5], dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 6:
                raise ValueError('Initial value must have size 6.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')



    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 6





    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 4



    def _rhs(self, y, t, alpha_0, alpha, beta, n):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(6)
        dy[0] = -y[0] + alpha / (1 + y[5]**n) + alpha_0
        dy[1] = -y[1] + alpha / (1 + y[3]**n) + alpha_0
        dy[2] = -y[2] + alpha / (1 + y[4]**n) + alpha_0
        dy[3] = -beta * (y[3] - y[0])
        dy[4] = -beta * (y[4] - y[1])
        dy[5] = -beta * (y[5] - y[2])
        return dy



    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        alpha_0, alpha, beta, n = parameters
        y = odeint(self._rhs, self._y0, times, (alpha_0, alpha, beta, n))
        return y[:, :]





    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        # Toni et al.:
        return np.array([1, 1000, 5, 2])


        # Figure 42 in book:
        #return np.array([0, 50, 0.2, 2])



    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        # Toni et al.:
        return np.linspace(0, 40, 400)



        # Figure 42 in book:
        #return np.linspace(0, 300, 600)


# Function to generate parameter combinations
def CombinationGenerator(n):
    # Define the parameter ranges
    alpha_0_range = (0, 5)
    alpha_range = (100, 5000)
    beta_range = (0, 10)
    n_range = (0, 10)

    # Set the number of combinations
    num_combinations = n

    # Use numpy to generate the combinations with uniform distribution
    alpha_0_values = np.random.uniform(alpha_0_range[0], alpha_0_range[1], num_combinations)
    alpha_values = np.random.uniform(alpha_range[0], alpha_range[1], num_combinations)
    beta_values = np.random.uniform(beta_range[0], beta_range[1], num_combinations)
    n_values = np.random.uniform(n_range[0], n_range[1], num_combinations)

    # Combine the values into a DataFrame
    combinations = pd.DataFrame({
        'alpha_0': alpha_0_values,
        'alpha': alpha_values,
        'beta': beta_values,
        'n': n_values
    })

    # Ensure uniqueness (though with uniform random generation, collisions are highly unlikely)
    combinations = combinations.drop_duplicates()

    # If there are not enough unique combinations, regenerate until we have enough
    while len(combinations) < num_combinations:
        additional_combinations = pd.DataFrame({
            'alpha_0': np.random.uniform(alpha_0_range[0], alpha_0_range[1], num_combinations - len(combinations)),
            'alpha': np.random.uniform(alpha_range[0], alpha_range[1], num_combinations - len(combinations)),
            'beta': np.random.uniform(beta_range[0], beta_range[1], num_combinations - len(combinations)),
            'n': np.random.uniform(n_range[0], n_range[1], num_combinations - len(combinations))
        })
        combinations = pd.concat([combinations, additional_combinations]).drop_duplicates()

    # Ensure we have exactly the desired number of unique combinations
    combinations = combinations.head(num_combinations)

    return combinations

# simulator function that takes the combination and the number of samples per combination and the suggested times
def simulator(combination,y,times=None):
    # If times is not provided, use the suggested times
    if times is None:
        times = RepressilatorModel().suggested_times()
    
    # Create the model
    model = RepressilatorModel(y0=y)
    # Simulate the model
    data = model.simulate(combination, times)
    
    # Add noise to the data
    #noise = np.random.normal(0, 0.1, data.shape)
    #data += noise
    # Return the data
    return data


# combination & number of samples
def generate_data(n, num_samples):
    # Generate the combinations
    combinations = CombinationGenerator(n)
    # transform combinations dataframe into a list of list
    combinations = combinations.values.tolist()
    # initial conditions
    if num_samples == 1:
        y = np.array([0, 0, 0, 2, 1, 3]).tolist()
    else:
        y = []
        for i in range(num_samples):
            y0 = np.random.uniform(0, 100, 6)
            y0 = y0.tolist()
            y.append(y0)
    
    # for each combination make a tuple of each combination and each member of y
    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))
    
    # shuffle the data
    np.random.shuffle(data)
    return data


if __name__ == '__main__':

    wd = os.getcwd()
    print(wd)
    
    parse = argparse.ArgumentParser(description="Perform basic mathematical operations.")
    parse.add_argument('--output_path', type=str, required=True, help='Path to save the output')
    parse.add_argument('--train_path', type=str, required=True, help='Path to save the training dataset')
    parse.add_argument('--valid_path', type=str, required=True, help='Path to save the validation dataset')
    parse.add_argument('--test_path', type=str, required=True, help='Path to save the test dataset')
    parse.add_argument('--n', type=int, required=True, help='Number of combinations')
    parse.add_argument('--num_samples', type=int, required=True, help='Number of samples per combination')
    args = parse.parse_args()

    
    # paths to save the datasets
    output_path = args.output_path
    train_data_path = os.path.join(wd,output_path,args.train_path)
    valid_data_path = os.path.join(wd,output_path,args.valid_path)
    test_data_path = os.path.join(wd,output_path,args.test_path)
    # number of combinations
    n = args.n
    # number of samples per combination
    num_samples = args.num_samples

    train_path = train_data_path
    valid_path = valid_data_path
    test_path = test_data_path
    # number of samples per combination
    times = np.linspace(0, 40, 1025)
    # generate the data
    combination = generate_data(n, num_samples)

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
            values = simulator(data[0],data[1],times)
            f.create_dataset(str(i), data=values)
    
    # create the validation dataset
    with h5py.File(valid_path, 'w') as f:
        for i, data in enumerate(valid_comb):
            values = simulator(data[0],data[1],times)
            f.create_dataset(str(i), data=values)

    # create the test dataset
    with h5py.File(test_path, 'w') as f:
        for i, data in enumerate(test_comb):
            values = simulator(data[0],data[1],times)
            f.create_dataset(str(i), data=values) 
