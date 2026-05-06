import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from ml.model import train_model
# TODO: add necessary import

# Used this website as a reference to learn how to do the 3 tests: https://medium.com/@ydmarinb/simplifying-unit-testing-in-machine-learning-with-python-df9b9c1a3300
# Test the training model function.
# Also used https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict for proper exporting syntax.
def test_training_model():
    X_train, y_train = make_classification(n_samples = 300, n_features = 5, random_state = 42)
    return_model = train_model(X_train,y_train)
    assert hasattr(return_model, "feature_importances_"), "Model needs to have feature_importances_ attribute"

# Test the inference function
def test_inference():
    X, y = make_classification(n_samples = 500, n_features = 4, random_state = 42)
    return_model = train_model(X,y) # Reused from last function
    return_inference = inference(return_model,X)
    assert set(return_inference) <= {0,1}, "The results should only be binary"

# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here

    pass
