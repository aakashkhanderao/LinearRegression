import requests
import pandas as pd
import scipy 
import numpy as np
import sys
from sklearn.model_selection import train_test_split


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"
def Intercept(a,b):
    intercept=np.mean(b)-Slope(a,b)*np.mean(a)
    return intercept
def Slope(a,b):
    n=len(a)
    two_sum=np.sum(a*b)
    sumX=np.sum(a)
    sumY=np.sum(b)
    sumX_2=np.sum(a**2)
    slope=(n*two_sum-sumX*sumY)/(n*sumX_2-(sumX)**2)
    return slope

def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    dataset1 = pd.read_csv("linreg_train.csv")
    dataset = dataset1.T
    test_size = 0.20
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size= test_size, random_state=1)
    intercept=Intercept(X_train,Y_train)
    slope=Slope(X_train,Y_train)
    


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
