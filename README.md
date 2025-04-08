💳 Credit Card Fraud Detection

This project focuses on identifying fraudulent credit card transactions using machine learning techniques.
The dataset used was obtained from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data),
and it contains real anonymized transactions from European cardholders in September 2013.


This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation.
Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.
Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.
Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning.
Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.


For this project, we used:
- Python for coding
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualizations
- Scikit-learn for training and evaluating the Logistic Regression model
- XGBoost for more advanced classification
- SHAP for model explainability (understanding feature importance)

  
📊 Exploratory Data Analysis (EDA)

Before building our models, we explored the dataset to understand its structure and characteristics.  
We looked at class distribution, duplicate values, feature ranges, and the distribution of transaction times and amounts. We also visualized PCA-transformed features (V1–V28) and analyzed fraud patterns over time.


We trained both Logistic Regression and XGBoost models, and handled the class imbalance using different techniques like `class_weight` and `scale_pos_weight`.

Our best results came from XGBoost, with:
- F1 Score: 0.90
- ROC AUC: 0.93
- AUPRC (Precision-Recall AUC): 0.81

We also used SHAP to visualize which features were most important in fraud detection.




