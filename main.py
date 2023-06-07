# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev & Usman Qureshi
# All rights reserved.
#
# This main.py file is part of the Advanced Optimization project (final) for
# the university course at Laurentian University.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# --------------------------------------------------------------------------------


# Section 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from nsga import run_nsga
from knn import knn
import matplotlib.pyplot as plt
import numpy as np


# Section 2: Load Datasets
sonar_data = pd.read_csv('dataset/sonar_dataset.csv')
musk_data = pd.read_csv('dataset/musk_dataset.csv')
datasets = {'SONAR': sonar_data, 'MUSK': musk_data}


# Section 3: Define Functions
def evaluate_individual(individual, X_train, y_train, X_test, y_test):
    """Evaluate each individual based on selected features and KNN performance."""
    selected_features = np.where(np.array(individual) == True)[0]
    return len(selected_features), knn(X_train, X_test, y_train, y_test, selected_features, k=5)


def hypervolume(population, obj1, obj2):
    """Calculate the hypervolume given a population and objectives."""
    sorted_population = sorted(population, key=lambda ind: (-ind.fitness.values[obj1], ind.fitness.values[obj2]))
    volume = 0.0
    a1 = sorted_population[0].fitness.values[obj1]
    a2 = sorted_population[0].fitness.values[obj2]
    for individual in sorted_population[1:]:
        b1 = individual.fitness.values[obj1]
        b2 = individual.fitness.values[obj2]
        volume += (a1 - b1) * a2
        a1 = b1
        a2 = b2
    return volume

def non_dominated_solutions(population):
    """Returns the non-dominated solutions from a population."""
    non_dominated = []
    for individual in population:
        is_dominated = False
        for other_individual in population:
            if (other_individual.fitness.values[0] < individual.fitness.values[0] and
                other_individual.fitness.values[1] <= individual.fitness.values[1]) or (
                other_individual.fitness.values[0] <= individual.fitness.values[0] and
                other_individual.fitness.values[1] < individual.fitness.values[1]):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append(individual)
    return non_dominated

# Section 4: Main Process - Training and Evaluation
print("\n==================== Training and Evaluation ====================\n")
for name, data in datasets.items():
    print(f"\n---- Processing Dataset: {name} ----")

    # Prepare data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Calculate classification error using all features
    all_features_error = knn(X_train, X_test, y_train, y_test, list(range(X.shape[1])))
    print(f'\nClassification error using all features for {name}: {all_features_error}\n')

    # Run NSGA-II algorithm
    initial_population, population, initial_fitness_values, final_population, final_fitness_values, logbook = run_nsga(X_train, y_train, X_test, y_test, k=5, Np=100, max_nfc=100 * 100)

    non_dominated_initial_population = non_dominated_solutions(initial_population)
    non_dominated_final_population = non_dominated_solutions(final_population)

    non_dominated_initial_fitness_values = np.array([ind.fitness.values for ind in non_dominated_initial_population])
    non_dominated_final_fitness_values = np.array([ind.fitness.values for ind in non_dominated_final_population])

    # Find the solution with minimum classification error
    min_error_index = np.argmin(final_fitness_values[:, 1])
    min_error_individual = final_population[min_error_index]
    selected_features = np.where(np.array(min_error_individual) == True)[0]

    # Print details of minimum error solution
    print(f'Number of features associated with MCE for {name}: {len(selected_features)}')
    print(f'The solution with minimum classification error for {name}: {selected_features}')

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(initial_fitness_values[:, 0], initial_fitness_values[:, 1], color='lightcoral', label='Feasible Solutions')
    non_dominated_initial_fitness_values = np.array([ind.fitness.values for ind in non_dominated_initial_population])
    non_dominated_initial_fitness_values = non_dominated_initial_fitness_values[non_dominated_initial_fitness_values[:,0].argsort()]  # Sort on the first objective
    plt.scatter(non_dominated_initial_fitness_values[:, 0], non_dominated_initial_fitness_values[:, 1], color='red', s=100, label='Non-Dominated Solutions')
    plt.plot(non_dominated_initial_fitness_values[:, 0], non_dominated_initial_fitness_values[:, 1], color='red')  # Plot lines
    plt.title(f'Initial Front for {name}', pad=20, fontsize=14)
    plt.xlabel('Number of Selected Features', labelpad=15, fontsize=12)
    plt.ylabel('Classification Error', labelpad=15, fontsize=12)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(final_fitness_values[:, 0], final_fitness_values[:, 1], color='lightgreen', label='Feasible Solutions')
    non_dominated_final_fitness_values = np.array([ind.fitness.values for ind in non_dominated_final_population])
    non_dominated_final_fitness_values = non_dominated_final_fitness_values[non_dominated_final_fitness_values[:,0].argsort()]  # Sort on the first objective
    plt.scatter(non_dominated_final_fitness_values[:, 0], non_dominated_final_fitness_values[:, 1], color='green', s=100, label='Non-Dominated Solutions')
    plt.plot(non_dominated_final_fitness_values[:, 0], non_dominated_final_fitness_values[:, 1], color='green')  # Plot lines
    plt.title(f'Final Front for {name}', pad=20, fontsize=14)
    plt.xlabel('Number of Selected Features', labelpad=15, fontsize=12)
    plt.ylabel('Classification Error', labelpad=15, fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Section 5: Hypervolume Calculation
print("\n==================== Hypervolume Calculation ====================\n")
hv_values = []
for name, data in datasets.items():
    print(f"\n---- Processing Dataset: {name} ----")

    # Prepare data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Run NSGA-II algorithm and calculate hypervolume for 15 runs
    for _ in range(15):
        initial_population, population, initial_fitness_values, final_population, final_fitness_values, logbook = run_nsga(X_train, y_train, X_test, y_test, k=5, Np=100, max_nfc=100 * 100)
        hv = hypervolume(final_population, 0, 1)
        hv_values.append(hv)

    # Calculate average hypervolume and print
    avg_hv = sum(hv_values) / len(hv_values)
    print(f'Average HV over 15 runs for {name}: {avg_hv}\n')
