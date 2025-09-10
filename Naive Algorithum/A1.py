# Q1) Decision Tree Classifier for Diabetes Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Load dataset (after downloading from Kaggle)
data = pd.read_csv("diabetes.csv")

# Features and Target
X = data.drop("Outcome", axis=1)  # independent variables
y = data["Outcome"]               # target variable

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Show rules of decision tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)