# ============================================================
# 📊 Project: Exam Mark Prediction using Linear Regression
# 📌 Type: Multiple Variable Linear Regression
# ============================================================

# ================================
# 🔹 Import Required Libraries
# ================================
import pandas as pd                      # For data handling
import numpy as np                       # For numerical operations
from sklearn.linear_model import LinearRegression   # ML model
from sklearn.model_selection import train_test_split # Train-test split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Evaluation metrics


# ================================
# 🔹 Load Dataset
# ================================
# Make sure 'data.csv' is in the same directory
dataset = pd.read_csv('data.csv')

# ================================
# 🔹 Dataset Overview
# ================================
print("Dataset Shape:", dataset.shape)   # Rows & Columns
print("\nFirst 5 Rows:\n", dataset.head())  # Preview data


# ================================
# 🔹 Handle Missing Values
# ================================
# Check columns with missing values
print("\nColumns with Missing Values:")
print(dataset.columns[dataset.isna().any()])

# Fill missing values in 'hours' column with mean
dataset['hours'] = dataset['hours'].fillna(dataset['hours'].mean())


# ================================
# 🔹 Split Features (X) & Target (Y)
# ================================
# X = Independent variables (all columns except last)
X = dataset.iloc[:, :-1].values
print("\nFeature Shape:", X.shape)

# Y = Dependent variable (last column - marks)
Y = dataset.iloc[:, -1].values


# ================================
# 🔹 Train-Test Split
# ================================
# 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)


# ================================
# 🔹 Model Training
# ================================
model = LinearRegression()
model.fit(X_train, y_train)


# ================================
# 🔹 Predictions
# ================================
y_pred = model.predict(X_test)


# ================================
# 🔹 Model Evaluation
# ================================
# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)   # Average error
mse = mean_squared_error(y_test, y_pred)    # Squared error
rmse = np.sqrt(mse)                         # Root error
r2 = r2_score(y_test, y_pred)               # Accuracy score

# Display results
print("\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# ================================
# 🔹 Custom Prediction (Optional)
# ================================
# Example input: [hours studied, sleep hours, previous score]
# a = [[9.2, 20, 0]]
# predicted_mark = model.predict(a)
# print("\nPredicted Exam Mark:", predicted_mark)