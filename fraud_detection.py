import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
file_path = r"E:\fraud_detection_using_data_analysis\Fraud_Detection_Dataset.csv"
df = pd.read_csv(file_path)

# Preview the dataset
print(df.head())
# Dataset Shape
print(f"Dataset Shape: {df.shape}")

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Summary Statistics
print("Summary Statistics:")
print(df.describe())
# Visualizing the distribution of transaction amounts
# Insight: The transaction amounts appear uniformly distributed,
# indicating that fraud occurs across all transaction values without specific peaks or clusters.
plt.figure(figsize=(10, 6))
sns.histplot(df["TransactionAmount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount (USD)")
plt.ylabel("Frequency")
plt.show()
# Visualizing the number of fraud cases by transaction type
# Insight: Certain transaction types are more prone to fraud than others.
# For example, online transactions have a higher number of fraud cases compared to cash transactions.
plt.figure(figsize=(8, 5))
sns.countplot(x="TransactionType", hue="FraudReported", data=df)
plt.title("Fraud Cases by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()
# Creating a heatmap to analyze correlations between features
# Insight: The heatmap shows that "TransactionAmount" has a moderate positive correlation with "FraudReported."
# However, most features do not have strong correlations with each other.
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
# The dataset contains 1,000 rows and 6 columns, with no missing values.
# Fraud has been reported in about 4.4% of the transactions.
# Model Training and Evaluation Code (newly added)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Feature set (X) and target variable (y)
X = df.drop("FraudReported", axis=1)  # Drop the target column from the dataset
y = df["FraudReported"]  # Target column

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)
print("Model training complete.")

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(rf_model, r"E:\fraud_detection_using_data_analysis\fraud_detection_model.pkl")
print("Model saved as fraud_detection_model.pkl")