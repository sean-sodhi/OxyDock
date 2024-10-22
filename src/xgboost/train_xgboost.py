# train_xgboost.py

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load the dataset
data = pd.read_csv('docking_data.csv')  # Replace with your actual data file

# Define features and target variable
# Ensure that the feature columns match those extracted in your C++ code
feature_columns = [
    'oxygen_level',
    'num_torsions',
    'gauss1', 
    'gauss2',
    'repulsion',
    'hydrophobic',
    'hydrogen_bonding'
    # Add other features as necessary
]
target_column = 'binding_affinity'

X = data[feature_columns]
y = data[target_column]

# Split the data into training and validation sets (e.g., 90% train, 10% validation)
train_size = int(0.9 * len(data))
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

# Create the XGBoost DMatrix objects
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Define training parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train the XGBoost model
num_rounds = 100
eval_list = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_rounds, evals=eval_list, early_stopping_rounds=10)

# Save the model in binary format
model.save_model('xgboost_model.bin')

# Optionally, save the model as a pickle file
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
