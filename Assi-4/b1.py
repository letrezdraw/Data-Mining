import pandas as pd
df=pd.read_csv("StudentsPerformance.csv")
print(df.shape)
print(df.head())
print(df.sample(5))
print(df.shape[1],df.columns)