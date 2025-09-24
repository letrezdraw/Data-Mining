import pandas as pd, numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    'Age': np.random.randint(18, 70, 50),
    'Annual Income': np.random.randint(15, 150, 50),
    'Spending Score': np.random.randint(1, 100, 50)
})

df.to_csv('Customer.csv', index=False)  # Save CSV

X = df.values
Z = linkage(X, method='ward')

dendrogram(Z)
plt.show()
