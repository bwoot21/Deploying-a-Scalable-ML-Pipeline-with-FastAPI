import pytest
from sklearn.datasets import make_classification
from ml.model import(train_model, inference)
from ml.data import apply_label

# Used this website from Udactiy mentor as a reference to learn how to do the 3 tests: https://medium.com/@ydmarinb/simplifying-unit-testing-in-machine-learning-with-python-df9b9c1a3300
# Test the training model function.
# Also used https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict for proper exporting syntax.
def test_training_model():
    """
    Create sample dataset with make_classification
    Run train_model function on that data and store in return_model
    Test with assert hassatr if Random Forest model has feature_importances_ attribute 
    """
    X_train, y_train = make_classification(n_samples = 300, n_features = 5, random_state = 42)
    return_model = train_model(X_train,y_train)
    assert hasattr(return_model, "feature_importances_"), "Model needs to have feature_importances_ attribute"

# Test the inference function
def test_inference():
    """
    Create sample dataset with make_classification
    Reuse train_model function to use for inference
    Run inference function on the model and data and store in return_inference
    Test with assert set to make sure the results are only binary
    """
    X, y = make_classification(n_samples = 500, n_features = 4, random_state = 42)
    return_model = train_model(X,y) # Reused from last function
    return_inference = inference(return_model,X)
    assert set(return_inference) <= {0,1}, "The results should only be binary"

# TODO: Test the apply label function
def test_apply_label():
    """
    Run function on two lists, each with 0 and 1
    Test with assert to see if results return the expected output.
    """
    zero = apply_label([0])
    one = apply_label([1])
    assert zero == "<=50K", "The result should return <=50K"
    assert one == ">50K", "The result should return >50K"
