from matplotlib import pyplot as plt

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, sep=",")
    df['Date'] = pd.to_datetime(df.Date)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[df['Temp'] > -20]
    y = pd.Series(df['DayOfYear'])
    X = pd.DataFrame(data=df)
    return X, y


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data("C:\\Users\\TESTUSER\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    Israel_subset = X.loc[X.Country == 'Israel']
    cmap = plt.cm.get_cmap('Spectral')
    color_dict = pd.Series({k: cmap(np.random.rand()) for k in Israel_subset['Year'].unique()})
    color_dict.name = 'color_dict'
    Israel_subset = pd.merge(Israel_subset, color_dict, how='left', left_on='Year', right_index=True)
    Israel_plot = Israel_subset.plot.scatter(x='DayOfYear', y='Temp', s=.15, c=Israel_subset['color_dict']);
    Israel_plot.set_title("Temp as a function of Day of year")
    Israel_plot.set_xlabel("Day of year")
    Israel_plot.set_ylabel("Temp")
    plt.show()
    ax = Israel_subset.groupby('Month').agg('std')['Temp'].plot.bar(x='Temp', y='Month', rot=0)
    ax.set_title("Standard deviation of temp for each month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temp std")
    plt.show()

    # Question 3 - Exploring differences between countries
    group_by_month = X.groupby(["Month", "Country"], as_index=False).agg({"Temp": ["std", 'mean']})
    fig = px.line(x=group_by_month.Month, y=group_by_month[("Temp", "mean")], color=group_by_month['Country'],
                  labels={"x": "Month", "y": " Mean Temp"},
                  error_y=group_by_month[("Temp", "std")],
                  title="Mean temp by month with std bars")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(Israel_subset['DayOfYear'], Israel_subset['Temp'], .75)
    res = []
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = poly.loss(train_X.to_numpy(), train_y.to_numpy())
        print("the loss of degree", k, "is", loss)
        res.append(loss)
    plt.bar(np.linspace(1, 10, 10), res)
    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.title("Test error as a function of degree")
    plt.show(block=False)
    plt.close()

    # Question 5 - Evaluating fitted model on different countries
    Jordan_subset = X.loc[X.Country == 'Jordan']
    SA_subset = X.loc[X.Country == 'South Africa']
    Netherlands_subset = X.loc[X.Country == 'The Netherlands']
    poly = PolynomialFitting(5)
    poly.fit(train_X.to_numpy(), train_y.to_numpy())
    Jordan_Loss = poly.loss(Jordan_subset['DayOfYear'], Jordan_subset['Temp'])
    SA_Loss = poly.loss(SA_subset['DayOfYear'], SA_subset['Temp'])
    Netherlands_Loss = poly.loss(Netherlands_subset['DayOfYear'], Netherlands_subset['Temp'])
    plt.bar(['Jordan', 'South Africa', 'The Netherlands'], [Jordan_Loss, SA_Loss, Netherlands_Loss])
    plt.xlabel("Country")
    plt.ylabel("Error")
    plt.title("Test error of Israel fitted modal on different countries")
    plt.show(block=False)
    plt.close()