import pandas as pd
from scipy import stats

from processing import DataProcessser
from reg_model import EPLSoccerRegression


class TTest:
    def __init__(self, model):
        self.model = model
        self.coefs = model.params
        self.pvalues = model.pvalues
        self.alpha = 0.05

    def test(self):
        for i, pvalue in enumerate(self.pvalues):
            if pvalue > self.alpha:
                print(
                    f"Coefficient for variable x{i+1} is not significant (p-value = {pvalue:.3f})"
                )
            else:
                print(f"Coefficient for variable x{i+1} is significant (p-value = {pvalue:.3f})")
class 

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

    ### 
    # T-test

    t_test = TTest(model)
    t_test.test()
