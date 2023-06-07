# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev & Usman Qureshi
# All rights reserved.
#
# This knn.py file is part of the Advanced Optimization project (final) for
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Section 2: Define KNN Function
def knn(X_train, X_test, y_train, y_test, selected_features, k=5):
    """
    This function applies the k-nearest neighbors (KNN) algorithm to a given set of training and test data.

    Parameters:
        X_train (array): The training dataset.
        X_test (array): The test dataset.
        y_train (array): The target variable for the training set.
        y_test (array): The target variable for the test set.
        selected_features (list): The list of selected features to be used in the model.
        k (int): The number of neighbors to consider (default is 5).

    Returns:
        error (float): The classification error of the model.
    """

    # Select features from datasets
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Initialize KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=k)

    # Fit the model
    classifier.fit(X_train_selected, y_train)

    # Predict the test set
    y_pred = classifier.predict(X_test_selected)

    # Compute classification error
    error = 1 - accuracy_score(y_test, y_pred)

    return error
