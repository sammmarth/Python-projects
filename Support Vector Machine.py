import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Load the data from the Excel file
data = pd.read_csv('BHU1.csv')
X = data.iloc[:, 1:-2]  # Features: Aspect columns
y = data['Maturity Level']  # Target: Maturity Level

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(C = 1.0, kernel = 'rbf', degree = 3, gamma = 'scale', coef0 = 0.0, shrinking = True,
                probability = False, tol = 0.001, cache_size =200, class_weight = None, verbose = False, max_iter = - 1,
                decision_function_shape = 'ovr', break_ties = False, random_state = 42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy and confusion matrix
precision = precision_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
print("Result Of SVM")
print("Accuracy:", accuracy)
print("Precision:", precision)
print(classification_report(y_test, y_pred))
# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(data['Total score'], data['Maturity Level'], c='blue', marker='o', label='Data Points')
plt.xlabel('Total Points')
plt.ylabel('Maturity Level')
plt.title('Sustainability Maturity Model for MSMEs')
plt.legend()
plt.grid(True)
plt.show()
