import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the data from Excel into a DataFrame
data = pd.read_csv('BHU1.csv')
# Step 3: Implement Neural Network
X = data.iloc[:, 1:-2]
y = data['Maturity Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
print("Neural Network Result")
print("Accuracy:" ,accuracy)
print("Precision:", precision)
print(classification_report(y_test, y_pred))


# Step 5: Create a graph
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Level')
plt.ylabel('Predicted Level')
plt.title('Actual vs. Predicted Level')
plt.show()
