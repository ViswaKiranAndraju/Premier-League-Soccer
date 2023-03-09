import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataProcessser:
    def __init__(self, data):
        self.data = data

    def run_processing(
        self,
        categorical_variables: list,
        independent_feature: str,
        target: str,
        test_size: float = 0.2,
    ):
        self.data = self.drop_missing_values()
        self.data = self.encoding_categorical_data(categorical_variables)
        self.data.drop(["PlayerName"], inplace=True, axis=1)
        X_train, X_test, y_train, y_test = self.split_data(independent_feature, target, test_size)
        return X_train, X_test, y_train, y_test

    def check_missing_values(self):
        return self.data.isnull().sum()

    def drop_missing_values(self):
        return self.data.dropna()

    def fill_missing_values(self, method: str = "mean"):
        if method == "mean":
            return self.data.fillna(self.data.mean())
        elif method == "median":
            return self.data.fillna(self.data.median())
        elif method == "mode":
            return self.data.fillna(self.data.mode())
        else:
            return self.data.fillna(0)

    def encoding_categorical_data(self, categorical_variables: list):
        for var in categorical_variables:
            self.data[var] = LabelEncoder().fit_transform(self.data[var])
        return self.data

    def split_data(self, independent_feature: str, target: str, test_size: float = 0.2):
        X = self.data[independent_feature]
        y = self.data[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test
