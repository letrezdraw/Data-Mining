import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
df=pd.read_csv("Iris.csv")
df=pd.get_dummies(df.drop(columns=["Id"]))
f=apriori(df,min_support=0.2,use_colnames=True)
r=association_rules(f,metric="confidence",min_threshold=0.5)
print(r)