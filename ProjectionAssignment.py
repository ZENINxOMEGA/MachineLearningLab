import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Step 1: Create sample dataset with two features (x and y)
data = {'x': [3, 5, 6, 9],
        'y': [10, 12, 15, 18]}

# Convert dictionary into a DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Step 2: Center the data by subtracting the mean (important for PCA)
# This ensures the data has zero mean, which is required for PCA.
X = df - df.mean()
print("\nMean Centered Data:\n", X)

# Step 3: Compute the covariance matrix of the centered data
# Covariance matrix shows the relationship between features (x, y).
COV_MATRIX = np.cov(X.T)   # Transpose required to get features in rows
print("\nCovariance Matrix:\n", COV_MATRIX)

# Step 4: Apply PCA (Principal Component Analysis)
# We reduce data to 1 principal component (n_components=1).
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Step 5: Print results of PCA
print("\nPCA Projection (data in new 1D space):\n", X_pca)

# Eigenvalues represent the variance captured by each principal component
print("\nEigenvalues (explained variance):\n", pca.explained_variance_)

# Eigenvectors represent the direction of principal components
print("\nEigenvectors (principal component directions):\n", pca.components_)

# Final projection of original data onto the principal component
print("\nFinal Projection onto Principal Component:\n", X_pca)
