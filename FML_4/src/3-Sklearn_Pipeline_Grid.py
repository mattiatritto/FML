import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

'''
This section contains all the variables and hyperparameters that we can change in order
to tune our model.

C: regularization parameter
penalty: regularization type
cv: number of folds used in cross-validation
scoring: measure used to evaluate the best model in Grid Search
'''
dataset_path = "../data/diabetes.csv"
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
target_feature = "Outcome"
random_state = 42
test_size = 0.2
C = [0.001, 0.01, 0.1, 1, 10, 100]
penalty = ['l1', 'l2']
cv=5
scoring="accuracy"




'''
Preprocessing steps
'''

dataset = pd.read_csv(dataset_path)
print(dataset.describe())
print(dataset.columns)

# Shuffling all samples to avoid group bias
#dataset = dataset.sample(frac=1).reset_index(drop=True) In order to make reproducible the results we have to add also here random_state
dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

# From the dataset we select only some features, and we convert them into numpy bidimensional arrays
X = dataset[selected_features].values
y = dataset[target_feature].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


numeric_features = list(range(X.shape[1]))
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'
    # By specifying remainder='passthrough', all remaining columns that were not specified in transformers, but present in the data passed to fit will be automatically passed through. This subset of columns is concatenated with the output of the transformers.
)

# Now we define the pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #('classifier', FML_3(random_state=random_state, C=1, penalty='l2')) C and penalty are not useful.
    ('classifier', LogisticRegression(random_state=random_state))
])





'''
Model creation
'''

param_grid = {
    'classifier__C': C,
    'classifier__penalty': penalty
}


grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, verbose=True)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
model = grid_search.best_estimator_
y_pred = model.predict(X_test)





'''
Model evaluation
'''

print(f'Best Hyperparameters: {best_params}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))