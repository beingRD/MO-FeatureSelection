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
