import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataset = pd.read_csv("datasets/diabetes.csv")
X = dataset.drop(columns=["Outcome"]).values
y = dataset["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
numeric_features = list(range(X.shape[1]))
preprocessor = ColumnTransformer(transformers=['num', numeric_transformer, numeric_features], remainder='passthrough')
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC())])

param_grid = {
    'C': [1, 5, 10],
    'tol': [0.01, 0.1, 0.001]
}

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=True)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
best_params = grid.best_params_
print(f'Accuracy: {accuracy_score(y_test, best_model.predict(X_test))}')