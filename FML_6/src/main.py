import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FML_6.src.ClassificationMetrics import ClassificationMetrics
from FML_6.src.ClassificationNeuralNetworks import ClassificationNeuralNetwork
from FML_6.src.RegressionMetrics import RegressionMetrics
from FML_6.src.RegressionNeuralNetwork import RegressionNeuralNetwork

# Variables to modify
dataset_path = "../data/houses_portaland_simple.csv"
selected_features = ['Size', 'Bedroom']
target_feature = "Price"
random_state = 42
test_size = 0.2



# Preprocessing steps
dataset = pd.read_csv(dataset_path)
dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
X = dataset[selected_features].values
y = dataset[target_feature].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)



# Neural Network creation
layers = [X.shape[1], 3, 1]
nn = RegressionNeuralNetwork(layers)
nn.fit(X_train_std, y_train)
preds = nn.predict(X_test_std)
nn.plot_loss()



# Evaluation of our NN
ev = RegressionMetrics(nn)
print(ev.compute_performance(X_test_std, y_test))





# Variables to modify
dataset_path = "../data/diabetes.csv"
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
target_feature = "Outcome"
random_state = 42
test_size = 0.2



# Preprocessing steps
dataset = pd.read_csv(dataset_path)
dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
X = dataset[selected_features].values
y = dataset[target_feature].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)



# Neural Network creation
layers = [X.shape[1], 3, 1]
nn = ClassificationNeuralNetwork(layers)
nn.fit(X_train_std, y_train)
preds = nn.predict(X_test_std)
nn.plot_loss()



# Evaluation of our NN
ev = ClassificationMetrics(nn)
print(ev.compute_performance(X_test_std, y_test))