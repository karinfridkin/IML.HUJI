from matplotlib import pyplot

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.io as pio
import plotly.express as px
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a

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
    fig.layout.update(
        title="Estimator error as a function of sample size",
        xaxis_title="Sample size",
        yaxis_title="Abs mean difference")
    fig.show()
<<<<<<< HEAD

    # Question 3 - Plotting Empirical PDF of fitted model
    plt.plot(random_arr, model.pdf(random_arr), 'o')
    plt.title("PDF as a function of random normal distribution samples")
    plt.xlabel("Sample")
    plt.ylabel("PDF")
    plt.savefig('PDF.png')
=======


    # Question 3 - Plotting Empirical PDF of fitted model
    plt.plot(random_arr, model.pdf(random_arr), 'o')
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    model = MultivariateGaussian()
    mu = (0, 0, 4, 0)
    cov = ((1, .2, 0, .5), (.2, 2, 0, 0), (0, 0, 1, 0), (.5, 0, 0, 1))
<<<<<<< HEAD
    random_arr2 = np.random.multivariate_normal(mu, cov, 1000)
    model.fit(random_arr2)
    print(model.fit)


    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    m = np.array(np.meshgrid(f1, np.zeros(1), f3, np.zeros(1))).reshape(4, -1)
    ll_applied_on_m = np.apply_along_axis(MultivariateGaussian.log_likelihood, 0, arr=m, cov=np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]), X=random_arr2)
    ll_applied_reshape = ll_applied_on_m.reshape(200, 200)
    plt.imshow(ll_applied_reshape, extent=[-10, 10, -10, 10])
    plt.title("log-liklihood as a function of mean in the form [f1, 0, f3,0]")
    plt.xlabel("f1")
    plt.ylabel("f3")
    plt.savefig('log_likelihood.png')
    plt.show()
=======
    random_arr = np.random.multivariate_normal(mu, cov, 1000)
    model.fit(random_arr)
    print(model.fit)

    # Question 5 - Likelihood evaluation
    a = np.linspace(-10, 10, 1000)
    b = np.linspace(-10, 10, 1000)
    for i in a:
        for j in b:
            fig = px.imshow(model.log_likelihood([a, 0, b, 0], model.cov_, random_arr))
    fig.show()
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a

    # Question 6 - Maximum likelihood
    print(f1[np.argmax(ll_applied_reshape, axis=0)][0])
    print(f3[np.argmax(ll_applied_reshape, axis=1)][0])


#if __name__ == '__main__':
#    np.random.seed(0)
#    test_univariate_gaussian()
#    test_multivariate_gaussian()
