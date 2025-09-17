
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

dataset = [
    ['Wine', 'Chips', 'Bread', 'Butter', 'Milk', 'Apple'],
    ['Wine', 'Chips', 'Butter', 'Milk'],
    ['Bread', 'Butter', 'Milk', 'Apple'],
    ['Bread', 'Milk', 'Apple'],
    ['Wine', 'Chips', 'Bread', 'Milk', 'Apple']
]
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(dataset), columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
with np.errstate(divide='ignore', invalid='ignore'):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
