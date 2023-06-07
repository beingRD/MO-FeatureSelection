# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev & Usman Qureshi
# All rights reserved.
#
# This nsga.py file is part of the Advanced Optimization project (final) for
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
import random
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from knn import knn


# Section 2: Define Evaluation Function
def evaluate(individual, X_train, y_train, X_test, y_test, k=5):
    """Evaluate each individual based on selected features and KNN performance."""
    selected_features = np.where(np.array(individual) == True)[0]
    return len(selected_features), knn(X_train, X_test, y_train, y_test, selected_features, k)


# Section 3: Define Multi-objective Fitness and Individual
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


# Section 4: Define NSGA-II Function
def run_nsga(X_train, y_train, X_test, y_test, k=5, Np=100, Cr=0.9, pm=0.01, max_nfc=None):
    """
    This function applies the Non-dominated Sorting Genetic Algorithm II (NSGA-II)
    to a given set of training and test data.
    """

    # Register necessary functions with toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, X_train.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation and genetic operations with toolbox
    toolbox.register("evaluate", evaluate, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, k=k)
    toolbox.register("mate", tools.cxUniform, indpb=Cr)
    toolbox.register("mutate", tools.mutFlipBit, indpb=pm)
    toolbox.register("select", tools.selNSGA2)

    # Initialize population and get initial fitness values
    population = toolbox.population(n=Np)
    initial_fitness_values = np.array([toolbox.evaluate(individual) for individual in population])
    initial_population = list(population)

    # Run the genetic algorithm
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=Np, lambda_=Np, cxpb=0.5, mutpb=0.1,
                                                    ngen=15, verbose=False)

    # Get final population and fitness values
    final_population = population
    final_fitness_values = np.array([toolbox.evaluate(individual) for individual in final_population])

    return initial_population, population, initial_fitness_values, final_population, final_fitness_values, logbook
