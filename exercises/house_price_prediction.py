import statistics

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"

import os


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename, sep=",")
    df.dropna(inplace=True)
    df = df[df['price'] > 0]
    df['is_new'] = 2015 - df['yr_built']
    y = pd.Series(df['price'])
    df = df.drop(['id', 'date', 'lat', 'long', 'waterfront', 'zipcode', 'yr_renovated', 'price'], axis=1)
    df = df[df.floors > 0]
    df = df[df.yr_built > 1800]
    df = df[df.sqft_living > 0]
    X = pd.DataFrame(data=df)
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        file_path = os.path.join(output_path, f'{feature}.png')
        plt.figure()
        np_feature = np.array(X[feature])
        np_y = np.array(y)
        cov = np.cov(np_y, np_feature)[0, 1]
        std_X = pd.Series(X[feature]).std()
        std_y = y.std()
        correlation = cov / (std_X * std_y)
        plt.ylabel('price')
        plt.xlabel(feature)
        plt.title(f'Price vs. {feature} [correlation={correlation:.3f}]')
        plt.scatter(X[feature], y)
        plt.savefig(file_path)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("C:\\Users\\TESTUSER\\IML.HUJI\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    res_mean=[]
    res_var=[]
    for percentage in np.linspace(0.1, 1, 91):
        res = []
        for _ in range(10):
            X_sample = X.sample(frac=(percentage))
            y_sample = y[X_sample.index]
            reg = LinearRegression()
            reg.fit(X_sample.to_numpy(), y_sample.to_numpy())
            res.append(reg.loss(test_X.to_numpy(), test_y.to_numpy()))
        res_mean.append(statistics.mean(res))
        res_var.append(np.var(res))
    plt.ylabel('MSE')
    plt.xlabel('percentage')
    plt.title('MSE vs. Percentage of train data used')
    perc = list(range(10, 101))

    plt.scatter(perc, res_mean)
    plt.show(block=False)

    plt.ylabel('Var')
    plt.xlabel('percentage')
    plt.title('Var vs. Percentage of train data used')
    perc = list(range(10, 101))
    print(res_var)

    plt.scatter(perc, res_var)
    plt.show(block=False)
    plt.close()
