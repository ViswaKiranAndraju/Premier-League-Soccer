import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm

# import seaborn as sns
from scipy import stats

from processing import DataProcessser


class EPLSoccerRegression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit_regression(self):
        X_train = sm.add_constant(self.X_train)
        model = sm.OLS(self.y_train, X_train).fit()
        return model

    def predict(self, model):
        X_test = sm.add_constant(self.X_test)
        return model.predict(X_test)

    def summarize(self, model):
        return model.summary()


def transform_predictors(X, transformation):
    """
    Transforms the predictor variables in the input array X using the specified
    transformation function.
    """
    # Check if the transformation function is valid
    if transformation not in ["log", "sqrt", "square"]:
        raise ValueError("Invalid transformation function.")

    # Apply the transformation to each column of the input array
    if transformation == "log":
        return np.log(X)
    elif transformation == "sqrt":
        return np.sqrt(X)
    elif transformation == "square":
        return np.square(X)


class EPLSoccerAssumptionsTest:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def linearity_using_visual_inspection(self):
        """
        Tests the linearity assumption by calculating the residuals and
        plotting them against the predictor variables. If the relationship
        between the predictor variables and the residuals is linear, then
        the assumption is satisfied.
        """
        X_pred = sm.add_constant(self.X)
        y_pred = self.model.predict(X_pred)
        residuals = self.y - y_pred

        plt.title(" Residuals vs. Cost")
        plt.xlabel("Cost", fontsize=15)
        plt.scatter(self.X, residuals)
        plt.show()

        plt.scatter(self.X, self.y)
        plt.xlabel("Cost")
        plt.ylabel("Score")
        plt.title("Score vs. Cost")
        plt.show()

    def linearity_test_using_coef(self):
        r, p = stats.pearsonr(self.X, self.y)

        # Test for a significant linear relationship
        if p < 0.05:
            print(p)
            print(
                "There is a significant linear relationship between the predictor and response variables."
            )
        else:
            print(
                "There is no significant linear relationship between the predictor and response variables."
            )
        return r, p

    def homoscedasticity(self):
        """
        You can plot the residuals against the predicted values and look for patterns in the scatter plot. If the variance of the residuals is constant,
        then you should see a random scatter of points around the center line, with no systematic patterns or trends.
        If the variance is not constant, then you should see a pattern or trend in the scatter plot, such as a funnel shape or an increase or decrease in
        the variance as the predicted values increase or decrease."""
        X_pred = sm.add_constant(self.X)
        y_pred = self.model.predict(X_pred)
        residuals = self.y - y_pred
        plt.subplots(figsize=(12, 6))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=self.X, y=residuals, alpha=0.5)
        plt.plot(np.repeat(0, self.X.max()), color="darkorange", linestyle="--")
        ax.spines["right"].set_visible(False)  # Removing the right spine
        ax.spines["top"].set_visible(False)  # Removing the top spine
        plt.title("Residuals")
        plt.show()

    def homoscedasticity_using_barlett_test(self):
        """
        If the p-value of the test is greater than 0.05, it indicates that there is not sufficient evidence to reject the null hypothesis and that the homoscedasticity assumption is satisfied. In this case, the code prints the message "The homoscedasticity assumption is satisfied."

        On the other hand, if the p-value of the test is less than 0.05, it indicates that there is sufficient evidence to reject the null hypothesis and that the homoscedasticity assumption is violated. In this case, the code prints the message "The homoscedasticity assumption is violated."
        """
        X_pred = sm.add_constant(self.X)
        y_pred = self.model.predict(X_pred)
        residuals = self.y - y_pred

        # Perform the Bartlett's test
        _, p = stats.bartlett(residuals, y_pred)

        # Test for homoscedasticity
        if p > 0.05:
            print("The homoscedasticity assumption is satisfied.")
        else:
            print(p)
            print("The homoscedasticity assumption is violated.")

    def normality(self):
        """
        Tests the normality assumption by calculating the residuals and
        plotting a histogram of the residuals. If the histogram is bell-shaped
        and the residuals are normally distributed, then the assumption is satisfied.
        """
        X_pred = sm.add_constant(self.X)
        y_pred = self.model.predict(X_pred)
        residuals = self.y - y_pred
        plt.hist(residuals)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

        scipy.stats.probplot(residuals, plot=plt)
        plt.show()

    def normality_using_shapiro_wilk_test(self):
        """
        If the p-value of the test is greater than 0.05, it indicates that there is not sufficient evidence to reject the null hypothesis and that the normality assumption is satisfied. In this case, the code prints the message "The normality assumption is satisfied."

        On the other hand, if the p-value of the test is less than 0.05, it indicates that there is sufficient evidence to reject the null hypothesis and that the normality assumption is violated. In this case, the code prints the message "The normality assumption is violated."
        """
        X_pred = sm.add_constant(self.X)
        y_pred = self.model.predict(X_pred)
        residuals = self.y - y_pred

        # Perform the Shapiro-Wilk test
        _, p = stats.shapiro(residuals)

        # Test for normality
        if p > 0.05:
            print("The normality assumption is satisfied.")
        else:
            print(p)
            print("The normality assumption is violated.")


class EPLSoccerPredict:
    def __init__(self, model, X, y) -> None:
        self.model = model
        self.X = X
        self.y = y

    def predict(self):
        X_pred = sm.add_constant(self.X)
        y_pred = self.model.predict(X_pred)
        return y_pred

    def plot_lr(self):
        plt.scatter(self.X, self.y)
        y_pred = self.predict()
        plt.plot(self.X, y_pred, "r")
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv("../epl_soccer_data.csv")
    data_processor = DataProcessser(data)
    X_train, X_test, y_train, y_test = data_processor.run_processing(
        categorical_variables=["Club"], independent_feature="Cost", target="Score"
    )
    # reset index to avoid error in statsmodels
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    regression = EPLSoccerRegression(X_train, X_test, y_train, y_test)
    model = regression.fit_regression()
    print(regression.summarize(model))

    predict = EPLSoccerPredict(model, X_test, y_test)
    predict.plot_lr()

    # Transform the predictor variables
    X_train_transformed = transform_predictors(X_train, "square")
    y_train_transformed = transform_predictors(y_train, "sqrt")
    # Test the assumptions
    assumptions_test = EPLSoccerAssumptionsTest(model, X_train, y_train)
    assumptions_test.linearity_using_visual_inspection()
    assumptions_test.linearity_test_using_coef()

    assumptions_test.homoscedasticity()
    assumptions_test.homoscedasticity_using_barlett_test()

    assumptions_test.normality()
    assumptions_test.normality_using_shapiro_wilk_test()
