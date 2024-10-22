# compile_model.py

import treelite
import treelite_runtime

# Load the XGBoost model
model = treelite.Model.load('xgboost_model.bin', model_format='xgboost')

# Compile the model into a shared library
model.export_lib(toolchain='gcc', libpath='xgboost_model.so', params={'parallel_comp': 4}, verbose=True)
