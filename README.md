# **Fraud Detection using Exploratory Data Analysis (EDA)**

This project focuses on detecting fraudulent transactions using data analysis and machine learning techniques. The primary goal is to analyze the dataset, extract meaningful insights, and build a predictive model to classify fraudulent and non-fraudulent transactions.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Machine Learning Model](#machine-learning-model)
6. [How to Run the Project](#how-to-run-the-project)
7. [Results](#results)
8. [License](#license)

---

## **Project Overview**
Fraud detection in financial transactions is crucial for maintaining the integrity of financial systems. This project demonstrates:
- Exploratory Data Analysis (EDA) to uncover hidden patterns in the data.
- Machine learning for predicting whether a transaction is fraudulent.
- Visualizations to better understand the data and its features.

---

## **Technologies Used**
- **Python**: Primary programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Matplotlib** & **Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning model training and evaluation.
- **Joblib**: Model saving and loading.

---

## **Dataset**
The dataset contains transaction records, including attributes like transaction amount, transaction type, and whether the transaction was fraudulent.

### **Key Features**:
- **TransactionType**: Type of transaction (e.g., online, cash).
- **TransactionAmount**: Amount of the transaction in USD.
- **FraudReported**: Target variable indicating whether the transaction was fraudulent (1 for fraud, 0 for non-fraud).

---

## **Exploratory Data Analysis (EDA)**
Key steps in the EDA process:
1. **Data Cleaning**:
   - Checked for missing values (none found).
   - Summary statistics of the dataset.
2. **Visualization**:
   - **Transaction Amount Distribution**: Histogram to show the spread of transaction amounts.
   - **Fraud Cases by Transaction Type**: Bar plot to analyze fraud occurrence based on transaction types.
   - **Correlation Heatmap**: Analyzed relationships between features.

---

## **Machine Learning Model**
- **Model Used**: Random Forest Classifier
- **Steps**:
  1. Preprocessed data (one-hot encoding for categorical features).
  2. Split the dataset into training (70%) and testing (30%) subsets.
  3. Trained a Random Forest Classifier on the training data.
  4. Evaluated the model using accuracy, confusion matrix, and classification report.

### **Model Performance**:
- Accuracy: ~90% (adjust this based on your results).
- Precision, Recall, F1-Score: Detailed in the classification report.

---

## **How to Run the Project**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnshumanCHauhan1/fraud-detection.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd fraud-detection
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Python script**:
   ```bash
   python fraud_detection.py
   ```
5. **View the saved model**: The trained model will be saved as `fraud_detection_model.pkl`.

---

## **Results**
- Fraud detection model trained successfully.
- EDA provided insights into key factors influencing fraudulent transactions.
- Model evaluation demonstrated high accuracy in detecting fraudulent activities.

---

## **License**
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute this project.
