# Loan-Approval-Risk-Prediction

**Loan Approval Prediction Using Random Forest**
Author: Gideon 
Tech Stack: Python, scikit-learn, pandas, seaborn, matplotlib

🔍 **Project Overview**
This project uses supervised machine learning to predict loan approval decisions based on borrower and loan attributes. The model was trained on a real-world dataset of personal loans issued in 2014. The goal is to identify key factors influencing loan approval and to build a robust classifier with high accuracy and generalizability.

🧠 **Problem Statement**
Can we predict whether a loan will be approved using borrower-related features such as credit score, income, and employment experience?

📂 **Dataset**
Source: Loan approval dataset (loan_data.csv)

Size: 13,500 records × 23 columns

Target Variable: loan_status (1 = approved, 0 = not approved)

**Key Features**:

person_income, loan_amnt, loan_int_rate, loan_percent_income

cb_person_cred_hist_length, credit_score

Categorical features: person_gender, person_home_ownership, loan_intent, education

🛠️ **Methodology**

**1. Data Cleaning**

Dropped rows with missing values using dropna()

Verified data types and encoded boolean columns properly

**2. Feature Engineering**

Converted categorical columns using pd.get_dummies()

Target variable: loan_status (binary classification)

**3. Modeling**

Used Random Forest Classifier with n_estimators=100, max_depth=12

Split data: train_test_split(X, y, test_size=0.3, random_state=42)

Scaled numeric features using StandardScaler for performance tuning

**4. Evaluation Metrics**

Accuracy: 92.7%

F1-score: 0.93 (weighted)

Cross-validation (k=5): 92.5% mean accuracy

Classification Report & Confusion Matrix used for evaluation

**5. Feature Importance**
Feature	Importance Score
previous_loan_defaults_on_file	0.22
loan_percent_income	0.16
loan_int_rate	0.15
person_income	0.12
loan_amnt	0.06

📈 **Results**
The model performed well on both majority and minority classes.

Previous defaults, income ratio, and interest rate were the top risk factors.

Random Forest showed better generalization than Logistic Regression.

**Classification Report**:

markdown
Copy
Edit
              precision    recall  f1-score   support
           0       0.94      0.97      0.95     10493
           1       0.89      0.77      0.82      3007
    Accuracy                           0.93     13500


📌**Key Learnings**

Working with imbalanced classes in binary classification

Feature importance and interpretability using Random Forest

The impact of preprocessing and scaling on model performance

Robust model evaluation using cross-validation

📚 **Future Improvements**

Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV

Experiment with other classifiers: XGBoost, Gradient Boosting

Use SMOTE or class weights to handle class imbalance better

Deploy the model using Streamlit for interactive use

