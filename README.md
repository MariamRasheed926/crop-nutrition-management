# Crop Nutrition Management

This project presents a machine learning-based system for analyzing crop nutrition data and supporting data-driven decisions related to crop productivity, fertilization, and irrigation management.

The model uses nutrient content, moisture, and yield-related measurements to identify relationships between crop conditions and productivity.

## Dataset

The dataset contains agricultural measurements related to crop nutrition and productivity, including:

* **Available Nitrogen (AvN%)**
* **Available Moisture (AvMoisture%)**
* **Average Yield Unit Weight (AvYieldUnitWeight (lb))**

The yield unit weight is used as the target variable for the model.

## Model

A **Random Forest Regressor** is used to model the relationship between nutrient and moisture conditions and crop yield.

The preprocessing pipeline includes:

* Converting numerical features and target values to the appropriate numeric format.
* Handling missing values using column mean imputation.
* Standardizing the input features.
* Training the model using the processed agricultural data.

## Performance

The model achieved a Mean Squared Error (MSE) of approximately:

**17.68**

This provides a baseline for evaluating the relationship between crop nutrition conditions and yield.

## Model Output

The trained model is saved as:

`nutrient_management_model.pkl`

The saved model can be reused for future predictions and further agricultural analysis.

## Development Tools

Python
Pandas
NumPy
Scikit-learn
Random Forest Regressor
Machine Learning

## Future Development

Future improvements may include incorporating additional soil, nutrient, irrigation, and environmental features to support more accurate crop nutrition and productivity management.
