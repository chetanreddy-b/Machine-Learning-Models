import math
import sys
from typing import List
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class FeatureReduction(object):

    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data: pd.DataFrame, target: pd.Series,
                          significance_level: float = 0.1) -> List[str]:
        """Forward Selection based on p-value significance"""
        selected_features = []
        remaining_features = list(data.columns)

        while remaining_features:
            p_values = {}            
            for feature in remaining_features:
                features_to_test = selected_features + [feature]
                X = sm.add_constant(data[features_to_test])
                model = sm.OLS(target, X).fit()
                p_values[feature] = model.pvalues[feature]

            best_feature = min(p_values, key=p_values.get)
            best_p_value = p_values[best_feature]

            if best_p_value < significance_level:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break

        return selected_features

    @staticmethod
    def backward_elimination(data: pd.DataFrame, target: pd.Series,
                             significance_level: float = 0.1) -> List[str]:
        """Backward Elimination based on p-value significance"""
        selected_features = list(data.columns)

        while selected_features:
            X = sm.add_constant(data[selected_features])
            model = sm.OLS(target, X).fit()
            p_values = model.pvalues[1:]  
            worst_feature = p_values.idxmax()
            worst_p_value = p_values[worst_feature]

            if worst_p_value >= significance_level:
                selected_features.remove(worst_feature)
            else:
                break

        return selected_features


    def evaluate_features(data: pd.DataFrame, y: pd.Series, features: list
        ) ->None:
        """
        PROVIDED TO STUDENTS

        Performs linear regression on the dataset only using the features discovered by feature reduction for each significance level.

        Args:
            data: (pandas data frame) contains the feature matrix
            y: (pandas series) output labels
            features: (python list) contains significant features. Each feature name is a string
        """
        print(f'Significant Features: {features}')
        data_curr_features = data[features]
        x_train, x_test, y_train, y_test = train_test_split(data_curr_features,
            y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        print(f'RMSE: {rmse}')
        print()
