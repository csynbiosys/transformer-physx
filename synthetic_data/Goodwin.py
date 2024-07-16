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



# Function to generate parameter combinations
def CombinationGenerator(n):
    # Define the parameter ranges
    k2 = (0, 10)
    k3 = (0, 10)
    m1 = (0, 1)
    m2 = (0, 0.1)
    m3 = (0, 1)

    # Set the number of combinations
    num_combinations = n

    # Use numpy to generate the combinations with uniform distribution
    k2_values = np.random.uniform(k2[0], k2[1], num_combinations)
    k3_values = np.random.uniform(k3[0], k3[1], num_combinations)
    m1_values = np.random.uniform(m1[0], m1[1], num_combinations)
    m2_values = np.random.uniform(m2[0], m2[1], num_combinations)
    m3_values = np.random.uniform(m3[0], m3[1], num_combinations)

    # Combine the values into a DataFrame
    combinations = pd.DataFrame({
        'k2': k2_values,
        'k3': k3_values,
        'm1': m1_values,
        'm2': m2_values,
        'm3': m3_values
    })

    # Ensure uniqueness (though with uniform random generation, collisions are highly unlikely)
    combinations = combinations.drop_duplicates()

    # If there are not enough unique combinations, regenerate until we have enough
    while len(combinations) < num_combinations:
        additional_combinations = pd.DataFrame({
            'k2': np.random.uniform(k2[0], k2[1], num_combinations - len(combinations)),
            'k2': np.random.uniform(k3[0], k3[1], num_combinations - len(combinations)),
            'm1': np.random.uniform(m1[0], m1[1], num_combinations - len(combinations)),
            'm2': np.random.uniform(m2[0], m2[1], num_combinations - len(combinations)),
            'm3': np.random.uniform(m3[0], m3[1], num_combinations - len(combinations))
        })
        combinations = pd.concat([combinations, additional_combinations]).drop_duplicates()

    # Ensure we have exactly the desired number of unique combinations
    combinations = combinations.head(num_combinations)

    return combinations

# simulator function that takes the combination and the number of samples per combination and the suggested times
def simulator(combination,y,times=None):
    # If times is not provided, use the suggested times
    if times is None:
        times = pints.toy.GoodwinOscillatorModel().suggested_times()
    
    # Create the model
    model = pints.toy.GoodwinOscillatorModel()
    model.set_initial_conditions(y)
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
        y = [0.0054, 0.053, 1.93]
    else:
        y = []
        for i in range(num_samples):
            #three random number as initial conditions the first from 0.001 to 0.01, the second from 0.01 to 0.1 and the third from 1 to 10 in a list
            y0 = np.random.uniform([0.001, 0.01, 1], [0.01, 0.1, 10], 3)
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
