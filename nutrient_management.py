import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your crop nutrient data
data = pd.read_csv('/kaggle/input/crop-nutrient-database/crops.csv')

# Print the column names to identify the correct features
print(data.columns)

# Replace 'feature1', 'feature2', 'feature3' with actual column names from your dataset
# For example, let's use 'AvN%(dry)', 'AvMoisture%', and 'AvYieldUnitWeight(lb)' as features
X = data[['AvN%(dry)', 'AvMoisture%', 'AvYieldUnitWeight(lb)']]  # Replace with your actual feature columns
y = data['AvYieldUnitWeight(lb)']  # Replace with your actual target column

# Clean the data by replacing non-numeric values with NaN and then filling them with the mean of the column
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# Ensure the target column is numeric and handle NaN values
y = pd.to_numeric(y, errors='coerce')
y.fillna(y.mean(), inplace=True)

# Check for any remaining NaN values
print(X.isna().sum())
print(y.isna().sum())
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model for future use
import joblib
joblib.dump(model, 'nutrient_management_model.pkl')






Index(['Crop', 'ScientificName', 'Symbol', 'NuContAvailable',
       'PlantPartHarvested', 'CropCategory', 'YieldUnit',
       'AvYieldUnitWeight(lb)', 'AvMoisture%', 'AvN%(dry)',
       ...
       'N%(wet)_M-FF', 'P%(wet)_M-FF', 'gP/100g(wet)_AgH8-9',
       'gP/100g(wet)_AgH8-12', 'gP/100g(wet)_B788', 'P%(wet)_M&L',
       'K%(wet)_M-FF', 'gK/100g(wet)_AgH8-9', 'gK/100g(wet)_AgH8-12',
       'gK/100g(wet)_B788'],
      dtype='object', length=161)
AvN%(dry)                0
AvMoisture%              0
AvYieldUnitWeight(lb)    0
dtype: int64
0
Mean Squared Error: 17.680802753277028
['nutrient_management_model.pkl']
