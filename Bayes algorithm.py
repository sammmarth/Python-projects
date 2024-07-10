#import supporting Librairies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Load data from csv
data = pd.read_csv('BHU1.csv')

# Split data into features (criteria) and labels (Level)
X = data.iloc[:, 1:-2]  # Features: Aspect columns
y = data['Maturity Level']  # Target: Maturity Level

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB(priors=None, var_smoothing=1e-09)

# Train the classifier on the training data
nb_classifier.fit(X_train, y_train)

# Predict the levels on the test data
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy and precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
print("Baye's Theorem Result")
print("Accuracy:" ,accuracy)
print("Precision:", precision)
print(classification_report(y_test, y_pred))

# Count the occurrences of each level (points) in the test set
points_counts = y_test.value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(points_counts, labels=points_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribution of Levels (Points)')
plt.show()
