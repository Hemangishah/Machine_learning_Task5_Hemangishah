import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset from your specified location
data_path = r'D:\Task\breast-cancer.csv'
data = pd.read_csv(data_path)

# Encode the 'diagnosis' column
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])  # 'M' -> 1, 'B' -> 0

# Separate features and target
X = data.drop(columns=['diagnosis', 'id'])  # Drop 'id' as it's not a useful feature
y = data['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Filter Method - SelectKBest with f_classif
select_kbest = SelectKBest(f_classif, k=10)
X_train_kbest = select_kbest.fit_transform(X_train, y_train)
X_test_kbest = select_kbest.transform(X_test)

# 2. Wrapper Method - Recursive Feature Elimination (RFE) with RandomForest
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# 3. Embedded Method - Lasso for feature importance
lasso = LassoCV()
lasso.fit(X_train, y_train)
X_train_lasso = X_train[:, lasso.coef_ != 0]
X_test_lasso = X_test[:, lasso.coef_ != 0]

# Train and Evaluate Model on Each Feature Set
model_kbest = RandomForestClassifier(random_state=42)
model_kbest.fit(X_train_kbest, y_train)
accuracy_kbest = accuracy_score(y_test, model_kbest.predict(X_test_kbest))

model_rfe = RandomForestClassifier(random_state=42)
model_rfe.fit(X_train_rfe, y_train)
accuracy_rfe = accuracy_score(y_test, model_rfe.predict(X_test_rfe))

model_lasso = RandomForestClassifier(random_state=42)
model_lasso.fit(X_train_lasso, y_train)
accuracy_lasso = accuracy_score(y_test, model_lasso.predict(X_test_lasso))

# Results
print("Accuracy with Filter Method (SelectKBest):", accuracy_kbest)
print("Accuracy with Wrapper Method (RFE):", accuracy_rfe)
print("Accuracy with Embedded Method (Lasso):", accuracy_lasso)
