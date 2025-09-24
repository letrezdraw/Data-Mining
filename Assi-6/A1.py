import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
num_samples = 1000
data = {
    "CUST_ID": [f"C{100000+i}" for i in range(num_samples)],
    "BALANCE": np.random.uniform(0, 5000, num_samples),
    "BALANCE_FREQUENCY": np.random.uniform(0, 1, num_samples),
    "PURCHASES": np.random.uniform(0, 10000, num_samples),
    "ONEOFF_PURCHASES": np.random.uniform(0, 5000, num_samples),
    "INSTALLMENTS_PURCHASES": np.random.uniform(0, 5000, num_samples),
    "CASH_ADVANCE": np.random.uniform(0, 10000, num_samples),
    "PURCHASES_FREQUENCY": np.random.uniform(0, 1, num_samples),
    "ONEOFF_PURCHASES_FREQUENCY": np.random.uniform(0, 1, num_samples),
    "PURCHASES_INSTALLMENTS_FREQUENCY": np.random.uniform(0, 1, num_samples),
    "CASH_ADVANCE_FREQUENCY": np.random.uniform(0, 1, num_samples),
    "CASH_ADVANCE_TRX": np.random.randint(0, 20, num_samples),
    "PURCHASES_TRX": np.random.randint(0, 100, num_samples),
    "CREDIT_LIMIT": np.random.uniform(1000, 20000, num_samples),
    "PAYMENTS": np.random.uniform(0, 20000, num_samples),
    "MINIMUM_PAYMENTS": np.random.uniform(0, 5000, num_samples),
    "PRC_FULL_PAYMENT": np.random.uniform(0, 1, num_samples),
    "TENURE": np.random.randint(6, 13, num_samples)
}
df = pd.DataFrame(data)
df.to_csv("CC_GENERAL.csv", index=False)
df = df.drop("CUST_ID", axis=1).fillna(df.mean(numeric_only=True))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=kmeans.labels_, palette='Set2')
plt.title("K-Means Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
df["Cluster"] = kmeans.labels_
df.to_csv("CC_CLUSTERED.csv", index=False)
