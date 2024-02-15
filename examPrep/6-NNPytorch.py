import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import optim



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = self.layer1(X)
        X = self.sigmoid(X)
        X = self.layer2(X)
        X = self.sigmoid(X)
        X = self.layer3(X)
        X = self.sigmoid(X)
        return X

def accuracy(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    return correct * 100 / len(y_true)



dataset = pd.read_csv("datasets/diabetes.csv")
dataset = dataset.values
X = dataset[:, 0:8]
y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

model = SimpleNN(input_size=X.shape[1], hidden_size=80, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2000):
    model.train()
    preds = model(X_train)
    preds = preds.squeeze()
    loss = criterion(preds, y_train)
    preds = torch.round(preds).float()
    acc = accuracy(preds, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model.eval()

    with torch.inference_mode():
        test_preds = model(X_test)
        test_preds = test_preds.squeeze()
        test_loss = criterion(test_preds, y_test)
        test_preds = torch.round(test_preds).float()
        test_acc = accuracy(test_preds, y_test)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")