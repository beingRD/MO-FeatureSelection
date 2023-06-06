# Feature Selection with NSGA-II and KNN
This project, developed as part of the Advanced Optimization course at Laurentian University, implements a feature selection mechanism using the Non-dominated Sorting Genetic Algorithm II (NSGA-II) and the k-nearest neighbors (KNN) algorithm. It is a collaborative effort by Rishabh Dev and Usman Qureshi.

The purpose of the project is to select the most relevant features from a dataset that can enhance the performance of a machine learning model. It includes an evaluation of the model's performance on the selected features, as well as an analysis of the hypervolume of the Pareto front to evaluate the quality of the solutions found by the NSGA-II algorithm.

Two datasets are used in this project: the Sonar dataset and the Musk dataset. Both datasets are placed inside the dataset directory.

# Installation
The project requires Python 3.7+ to run and assumes that pandas, numpy, sklearn, matplotlib, and deap libraries are installed.

If you haven't installed them, you can do so using pip:


```python
pip install pandas numpy scikit-learn matplotlib deap
```
First, place your datasets in the dataset directory. The default datasets are 'sonar' and 'musk', but you can replace them with any datasets of your choice.

Run the main.py script to start the feature selection process:

```python
python main.py
```
The script will output the classification error using all features, the number of features associated with minimum classification error, and a visual representation of the initial and final Pareto fronts for each dataset.

Finally, it will calculate and output the average hypervolume over 15 runs for each dataset.

# Files
The project includes the following files:

- main.py: The main script that loads the datasets, runs the NSGA-II algorithm, performs feature selection, evaluates the solutions, and calculates the hypervolume.
- nsga.py: Contains the implementation of the NSGA-II algorithm.
- knn.py: Contains the implementation of the KNN classifier.

# Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

# License
The code in this project is licensed under the MIT license.

# Authors
- Rishabh Dev
- Usman Qureshi
