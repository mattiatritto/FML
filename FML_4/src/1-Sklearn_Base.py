import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

'''
This section contains all the variables and hyperparameters that we can change in order
to tune our model.
'''
dataset_path = "../data/diabetes.csv"
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
target_feature = "Outcome"
random_state = 42





'''
Preprocessing steps
'''

dataset = pd.read_csv(dataset_path)
print(dataset.describe())
print(dataset.columns)

#dataset = dataset.sample(frac=1).reset_index(drop=True) In order to make reproducible the results we have to add also here random_state
dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

# From the dataset we select only some features, and we convert them into numpy bidimensional arrays
X = dataset[selected_features].values
y = dataset[target_feature].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# In order to standardize the features, we use the StandardScaler()
scaler = StandardScaler()
# If we want to be more robust against outlier, we can use RobustScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)





'''
MODEL CREATION
'''

model = LogisticRegression(random_state=random_state)
model = model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)





'''
MODEL EVALUATION
'''


print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))







