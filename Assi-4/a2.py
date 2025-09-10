import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
df=pd.read_csv("Groceries_dataset.csv")
basket=df.groupby(['Member_number','Date'])['itemDescription'].apply(list)
oht=pd.get_dummies(basket.apply(pd.Series).stack()).sum(level=0)
f=apriori(oht,min_support=0.01,use_colnames=True)
r=association_rules(f,metric="confidence",min_threshold=0.3)
print(r[['antecedents','consequents','support','confidence']])