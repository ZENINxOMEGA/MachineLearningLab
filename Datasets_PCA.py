# ------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
# 2. LOAD AND EXPLORE THE DATASET
# ------------------------------------------------------------------

# Load the built-in digits dataset
digits_dataset = load_digits()

# Separate features (X) and target (Y)
X = digits_dataset.data
Y = digits_dataset.target

print("--- Data Exploration ---")
print("Original feature shape (X):", X.shape) # (1797 images, 64 pixels each)
print("Original target shape (Y):", Y.shape)

# ------------------------------------------------------------------
# 3. DATA PREPROCESSING AND SPLITTING
# ------------------------------------------------------------------

# Step 3.1: Scale the features
# This is crucial for PCA to work effectively.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3.2: Split data into training and testing sets
# We create one consistent split to test both our models fairly.
# random_state=42 ensures the split is the same every time for reproducibility.
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

print("\n--- Data Preprocessing & Splitting ---")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------------
# 4. MODEL 1: LOGISTIC REGRESSION ON FULL DATA (64 FEATURES)
# ------------------------------------------------------------------

# Initialize and train the model on the full 64 features.
model_full = LogisticRegression(max_iter=1000)
model_full.fit(X_train, Y_train)

# Evaluate the model's accuracy
accuracy_full = model_full.score(X_test, Y_test)
print(f"\n--- Model 1: Using all {X_train.shape[1]} features ---")
print(f"Accuracy: {accuracy_full * 100:.2f}%")

# Generate predictions to create the confusion matrix
Y_pred_full = model_full.predict(X_test)
cm_full = confusion_matrix(Y_test, Y_pred_full)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Model 1 (Full Features)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ------------------------------------------------------------------
# 5. APPLY PCA FOR DIMENSIONALITY REDUCTION
# ------------------------------------------------------------------

# Initialize PCA to keep components that explain 95% of the variance.
pca = PCA(n_components=0.95)

# Fit PCA on the TRAINING data and transform both train and test data.
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\n--- PCA Dimensionality Reduction ---")
print("Original number of features:", pca.n_features_in_)
print("Reduced number of features:", pca.n_components_)

# ------------------------------------------------------------------
# 6. MODEL 2: LOGISTIC REGRESSION ON PCA-REDUCED DATA
# ------------------------------------------------------------------

# Initialize and train a new model on the dimensionally reduced data.
model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_train_pca, Y_train)

# Evaluate the model's accuracy on the PCA-transformed test set
accuracy_pca = model_pca.score(X_test_pca, Y_test)
print(f"\n--- Model 2: Using {pca.n_components_} PCA features ---")
print(f"Accuracy: {accuracy_pca * 100:.2f}%")

# Generate predictions to create the confusion matrix
Y_pred_pca = model_pca.predict(X_test_pca)
cm_pca = confusion_matrix(Y_test, Y_pred_pca)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Reds')
plt.title(f'Confusion Matrix for Model 2 ({pca.n_components_} PCA Features)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()