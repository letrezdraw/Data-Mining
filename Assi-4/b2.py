import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
df=pd.read_csv("groceries.csv")
oht=pd.get_dummies(df.stack()).sum(level=0)
f=apriori(oht,min_support=0.004,min_confidence=0.2,min_lift=3,min_len=2,use_colnames=True)
r=association_rules(f,metric="lift",min_threshold=3)
print(r[['antecedents','consequents','support','confidence','lift']])