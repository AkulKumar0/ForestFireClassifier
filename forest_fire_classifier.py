import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier


#Preprocessing
data = pd.read_csv(r'C:\Users\rajku\PycharmProjects\pythonProject10\forest.csv')

def categorize_burned_area(area):
    if area == 0:
        return "low"
    elif area <= 10:
        return "low"
    elif area <= 100:
        return "medium"
    else:
        return "high"

data['burned_area_category'] = data['area'].apply(categorize_burned_area)

data.drop(columns=['area'], inplace=True)

X = data.drop(columns=['burned_area_category'])
y = data['burned_area_category']

# Step 3: Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Preprocessing - Convert Categorical Features with One-Hot Encoding
categorical_columns = ['month', 'day']

X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Make sure that the test set columns match the training set columns
missing_columns = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for column in missing_columns:
    X_test_encoded[column] = 0  # Add missing columns to the test set with default values

# Reorder test set columns to match the order in the training set
X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Step 5: Preprocessing - Handling Imbalanced Classes with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

# Step 6: Initialize and Train the Models
models = {
    'Decision Trees': DecisionTreeClassifier(random_state=42),
    'Random Forests': RandomForestClassifier(random_state=42),
    'Support Vector Machines': SVC(random_state=42),

}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    trained_models[name] = model

# Step 7: Make Predictions and Evaluate Models
for name, model in trained_models.items():
    y_pred = model.predict(X_test_encoded)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calculate ROC AUC for each class separately
    roc_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), average='macro')

    print(f'Model: {name}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC-ROC Score: {roc_auc:.2f}')
    print('-' * 40)







