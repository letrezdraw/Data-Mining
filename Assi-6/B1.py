import numpy as np

# Generate synthetic data: 100 points, 2D
np.random.seed(0)
X = np.vstack((np.random.randn(50,2)+[5,5], np.random.randn(50,2)+[0,0]))

k = 2
centers = X[np.random.choice(len(X), k, False)]
for _ in range(10):
    labels = np.argmin(((X[:,None]-centers)**2).sum(2), axis=1)
    centers = np.array([X[labels==i].mean(0) if len(X[labels==i]) else centers[i] for i in range(k)])

print("Centers:\n", centers)
print("Labels:\n", labels)
