# Section 1: Import Required Libraries
import random
from deap import base, creator, tools, algorithms
import numpy as np
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
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=Np, lambda_=Np, cxpb=0.5, mutpb=0.1, ngen=15, verbose=False)

    # Get final population and fitness values
    final_population = population
    final_fitness_values = np.array([toolbox.evaluate(individual) for individual in final_population])

    return initial_population, population, initial_fitness_values, final_population, final_fitness_values, logbook

