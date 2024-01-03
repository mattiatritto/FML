import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

'''
The following section is dedicated to all the variables and hyperparameters 
that we can change to tune our model.

random_state: a number that makes our experiments reproducible. This means that all the random functions used in this code will also produce the same seed to make random numbers.
criterion: represents the function to measure the quality of a split
min_samples_leaf: sets the minimum number of samples required to be at a leaf node
max_depth: limits the maximum depth of the decision tree
cv: defines the number of folds in the Grid Search
scoring: defines the criterion to evaluate the best model across the multiple ones

<< min_samples_leaf & >> max_depth --> more complexity (overfitting)
>> min_samples_leaf & << max_depth --> less complexity (underfitting)
'''

data_path = "../data/diabetes.csv"
random_state = 42
test_size = 0.2
# Instead of specifying just a single hyperparameter, we specify a list of hyperparameters for the Grid Search
criterion = ['entropy']
min_samples_leaf = [2, 5, 10, 20]
max_depth = [5, 10, 20]
# New variables in comparison to the simple code
cv=5
scoring = 'accuracy'



'''
PRE-PROCESSING STEPS
'''
# Load the dataset
dataset = pd.read_csv(data_path)

# Divide features and target variables, transforming them into numpy matrices
X = dataset.drop(['Outcome'], axis=1).values
y = dataset['Outcome'].values

# Split the dataset into training and test sets through a simple hold-out strategy. The parameter stratify allows to stratify data based on target feature.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, stratify=y)

'''
Do we need data normalization if we use a decision tree?

NO, because the meaning of scaling data through normalization is to ensure that a specific feature is not prioritized
over another due to the feature values magnitude (it is a Gradient Descent consequence). In the case of decision trees,
the algorithm makes decisions based on comparisons of feature values at different nodes of the tree, and the relative 
ordering of the values is what matters, not their absolute scale. Indeed, we just split data.
'''




'''
GRID SEARCH
'''

param_grid = {
    'criterion': criterion,
    'min_samples_leaf': min_samples_leaf,
    'max_depth': max_depth
}
model = DecisionTreeClassifier(random_state=random_state)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose=True)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
model = grid_search.best_estimator_
y_pred = model.predict(X_test)
print(f'Parameters of the best model used: {best_params}')




'''
MODEL EVALUATION AND VISUALIZATION
'''

y_pred_train = model.predict(X_train)
print(f'Accuracy on training set: {accuracy_score(y_train, y_pred_train)}')
print(f'Accuracy on test set: {accuracy_score(y_test, y_pred)}')
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=dataset.columns, filled=True)
#plt.show()
plt.savefig(f'../images/{best_params["criterion"]}_{best_params["min_samples_leaf"]}_{best_params["max_depth"]}.png', format='png',
            bbox_inches="tight", dpi=199)