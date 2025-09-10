import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("shows.csv")
data["Nationality"] = LabelEncoder().fit_transform(data["Nationality"])
data["Go"] = LabelEncoder().fit_transform(data["Go"])
X, y = data[["Age", "Experience", "Rank", "Nationality"]], data["Go"]
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X, y)
le_nat = LabelEncoder().fit(data["Nationality"])
print("Prediction:", LabelEncoder().fit(data["Go"]).inverse_transform(clf.predict([[40, 10, 7, le_nat.transform(["USA"])[0]]]))[0])
