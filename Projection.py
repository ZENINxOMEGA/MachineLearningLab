import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Step 1: Create dataset with two features (x and y)
data = {'x': [4, 6, 8, 9],
        'y': [10, 20, 30, 40]}

# Convert dictionary to DataFrame for better handling
df = pd.DataFrame(data)
print("Original Data:\n", df)

# Step 2: Center the data (subtract mean of each column)
# This ensures the dataset has zero mean, which is required before PCA.
X = df - df.mean()
print("\nMean Centered Data:\n", X)

# Step 3: Compute Covariance Matrix
# Covariance shows how strongly two variables vary together.
# np.cov requires rows as features, so we use transpose (X.T).
COV_MATRIX = np.cov(X.T)
print("\nCovariance Matrix:\n", COV_MATRIX)

# Step 4: Apply PCA to reduce to 1 principal component
# PCA finds new axes (principal components) that maximize variance.
pca = PCA(n_components=1)   # keep only 1 component
X_pca = pca.fit_transform(X)

# Step 5: Print PCA results
print("\nPCA Projection (data in 1D space):\n", X_pca)

# Eigenvalues = Variance explained by each principal component
print("\nEigenvalues (explained variance):\n", pca.explained_variance_)

# Eigenvectors = Principal component directions (unit vectors)
print("\nEigenvectors (principal component directions):\n", pca.components_)

# Step 6: Final Projection
# Original data is projected onto the principal component (reduced dimension)
print("\nFinal Projection onto Principal Component:\n", X_pca)
