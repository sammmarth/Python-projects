import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
data = pd.read_csv('BHU1.csv')
X = data.iloc[:, 1:-2]
y = data['Maturity Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                             max_features = 'sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                             n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
print("Result Of Random Forest")
print("Accuracy:", accuracy)
print("Precision:", precision)
print(classification_report(y_test, y_pred))

cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Level")
plt.ylabel("True Level")
plt.show()
