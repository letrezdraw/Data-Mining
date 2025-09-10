# Q2) Decision Tree Classifier for shows.csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("shows.csv")

# Convert categorical data to numbers
le_nationality = LabelEncoder()
data["Nationality"] = le_nationality.fit_transform(data["Nationality"])

le_go = LabelEncoder()
data["Go"] = le_go.fit_transform(data["Go"])   # YES=1, NO=0

# Features & Target
X = data[["Age", "Experience", "Rank", "Nationality"]]
y = data["Go"]

# Train model
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X, y)

# Predict for: 40 years old American, 10 years exp, Rank=7
test_input = [[40, 10, 7, le_nationality.transform(["USA"])[0]]]
prediction = clf.predict(test_input)

print("Prediction:", le_go.inverse_transform(prediction)[0])