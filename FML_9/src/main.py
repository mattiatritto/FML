import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from Evaluation import Evaluation



# Settings to be adjusted
path_dataset = "../data/diabetes.csv"
target_feature = "Outcome"
selected_features = ["Pregnancies", "Glucose", "BloodPressure"]
training_size = 0.9
random_state = 42



# PREPROCESSING STEPS

# Read the dataset
dataset = pd.read_csv(path_dataset)

# Divide between features and target variables
y = dataset[target_feature].values
X = dataset[selected_features].values

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, random_state=random_state)

# Standard Scaler is for normalizing data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# MODEL CREATION AND TRAINING EVALUATION
model = SVC(C=5, random_state=random_state)
model = model.fit(X_train, y_train)
ev = Evaluation(model)
print(ev.compute_performance(X_train, y_train))



# TESTING OUR MODEL ON UNSEEN DATA
ev = Evaluation(model)
print(ev.compute_performance(X_test, y_test))