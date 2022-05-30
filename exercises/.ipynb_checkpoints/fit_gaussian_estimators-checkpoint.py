from matplotlib import pyplot

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"
from scipy.stats import norm
import matplotlib.pyplot as plt


def test_univariate_gaussian():
    model = UnivariateGaussian()
    random_arr = np.random.normal(10, 1, 1000)
    model.fit(random_arr)

    print(model.mu_, model.var_)

    # Question 2 - Empirically showing sample mean is consistent
    abs_distance = []
    for i in range(10, 1000, 10):
        abs_distance.append(abs(model.mu_ - np.mean(random_arr[0:i])))

    fig = px.scatter(x=list(range(10, 1000, 10)), y=abs_distance)
    fig.update_layout(
        title="Estimator error as a function of sample size",
        xaxis_title="Sample size",
        yaxis_title="Abs mean difference")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    model.fit(random_arr)
    fig = px.scatter(x=list(range(0, 1000)), y=model.pdf(random_arr))
    fig.update_layout(
        title="PDF",
        xaxis_title="num",
        yaxis_title="probability")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    model = MultivariateGaussian()
    mu = (0, 0, 4, 0)
    cov = ((1, .2, 0, .5), (.2, 2, 0, 0), (0, 0, 1, 0), (.5, 0, 0, 1))
    random_arr = np.random.multivariate_normal(mu, cov, 1000)
    model.fit(random_arr)
    print(model.fit)

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
