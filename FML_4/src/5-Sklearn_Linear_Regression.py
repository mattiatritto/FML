import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression



class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound = []
        self.upper_bound = []

    def outlier_detector(self, X):
        # Calculate quartiles
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)

        # Calculate IQR (Interquartile Range)
        iqr = q3 - q1

        # Calculate lower and upper bounds to identify outliers
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        # Initialize lower and upper bounds
        self.lower_bound = []
        self.upper_bound = []

        # Apply the outlier_detector function along axis 0 (columns)
        np.apply_along_axis(self.outlier_detector, axis=0, arr=X)

        return self

    def transform(self, X, y=None):
        # Copy the input array to avoid unwanted changes
        X = np.copy(X)

        # Iterate over all columns
        for i in range(X.shape[1]):
            x = X[:, i]

            # Masks to identify outliers
            lower_mask = x < self.lower_bound[i]
            upper_mask = x > self.upper_bound[i]

            # Set values that are considered outliers to NaN
            x[lower_mask | upper_mask] = np.nan

            # Assign the transformed column back to the original array
            X[:, i] = x

        # Impute NaN values with the mean
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        return X


# Variables to modify
filepath_dataset = "../data/houses_portaland_simple.csv"
random_state = 42



# Read the dataset, shuffle the examples and select the features we want to use for our model
dataset = pd.read_csv(filepath_dataset)
dataset = dataset.sample(frac=1).reset_index(drop=True)


selected_features = ['Size', 'Bedroom']
X = np.float_(dataset[selected_features].values)
y = np.float_(dataset['Price'].values)


# Here we split the dataset into 80% Training and 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)


# Here we have our pipeline

numeric_features = list(range(X.shape[1])) # This is for using integer indices
numeric_transformer = Pipeline(steps=[
    ('outlier_remover', OutlierRemover()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

"""
1) Numeric_transformer processed features replace 
the original numerical features in the pipeline, retaining only the modifications.
2) remainder = passthrough, the features not involved in the transformations 
are included in the output without undergoing any modification."""

# The preprocessor manages to remove outliers, imputation and standardization of numerical features
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


param_grid = [{
    "regressor__fit_intercept": [True, False]
}]

# Here we have the grid search
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=20, scoring='neg_root_mean_squared_error', verbose=True)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# Compute the metrics that are related to the specific model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
#print(f'Best Hyperparameters: {best_params}')
print(f'MSE with Best Model: {mse:.2f}')
print('R2 score with best model:\n', r2)