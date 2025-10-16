# ------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------------------------------------------------------
# 2. LOAD AND ANALYZE THE DATA
# ------------------------------------------------------------------

# Load the Iris dataset
iris = load_iris()

# Create a pandas DataFrame for easier analysis
# The DataFrame will have the feature data and a column for the target species
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# Display basic information about the dataset
print("--- Data Analysis ---")
print("First 5 rows of the dataset:")
print(iris_df.head())
print("\nDataset Information:")
iris_df.info()
print("\nStatistical Summary:")
print(iris_df.describe())
print("\nSpecies Count:")
print(iris_df['species'].value_counts())

# ------------------------------------------------------------------
# 3. VISUALIZE THE DATA
# ------------------------------------------------------------------

print("\n--- Generating Visualizations ---")

# Use a pair plot to visualize relationships between all feature pairs
# The 'hue' parameter colors the points by species, making it easy to see separation
sns.pairplot(iris_df, hue='species', palette='viridis', markers=["o", "s", "D"])
plt.suptitle('Pair Plot of Iris Dataset Features', y=1.02)
plt.show()


# Use violin plots to see the distribution of each feature for each species
plt.figure(figsize=(12, 10))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.violinplot(x='species', y=feature, data=iris_df, palette='plasma')
plt.suptitle('Feature Distribution by Species', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ------------------------------------------------------------------
# 4. PREPARE DATA FOR MODELING
# ------------------------------------------------------------------

# Separate features (X) and target (y)
X = iris.data
y = iris.target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
# This helps the model converge faster and perform better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Data Splitting ---")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# ------------------------------------------------------------------
# 5. TRAIN AND EVALUATE THE MODEL
# ------------------------------------------------------------------

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)
print("\n--- Model Training Complete ---")

# Make predictions on the scaled test data
y_pred = model.predict(X_test_scaled)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------------------------
# 6. VISUALIZE THE CONFUSION MATRIX
# ------------------------------------------------------------------

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()