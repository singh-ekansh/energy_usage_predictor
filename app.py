import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add style for the page
page_bg = """
<style>
body {
    background-image: url('matthew-henry-yETqkLnhsUI-unsplash.jpg'); 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
.header {
    font-size: 30px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(0, 0, 0, 0.8);
    color: white;
    text-align: center;
    padding: 10px;
}
</style>
"""

# Background and header styling
st.markdown(page_bg, unsafe_allow_html=True)

# Header
st.markdown('<div class="header">Electricity Bill Prediction App</div>', unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    file_path = 'electricity_bill_dataset.csv'
    return pd.read_csv(file_path)

data = load_data()

# Split data in numerical and categorical column
categorical_columns = ['City', 'Company']
numerical_columns = ['Fan', 'Refrigerator', 'AirConditioner', 'Television',
                     'Monitor', 'MotorPump', 'Month', 'MonthlyHours', 'TariffRate']

# Preprocessing: Scale numerical features and convert categorical data into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Separate features (X) and target (y)
X = data.drop(columns=['ElectricityBill'])
y = data['ElectricityBill']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying preprocessing on the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Select algorithm
algorithm = st.selectbox(
    "Select Algorithm for Prediction",
    ["Linear Regression", "Ridge Regression", "Lasso Regression",
     "Support Vector Regression", "Decision Tree", "Random Forest",
     "XGBoost", "Artificial Neural Network"]
)

# Define models based on selection
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Support Vector Regression": SVR(kernel='rbf', C=100, epsilon=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    "Artificial Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# Train the selected model
selected_model = models[algorithm]
selected_model.fit(X_train_preprocessed, y_train)

# Predict using the selected model
y_pred = selected_model.predict(X_test_preprocessed)

# Calculate the accuracy metrics for the selected model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Show metrics
st.subheader(f"Model Accuracy ({algorithm})")
st.write(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
st.write(f"Mean Squared Error (MSE): ₹{mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Interface
st.write("Provide your inputs below to predict the monthly electricity bill:")

# Take user inputs
fan = st.number_input("Number of Fans", min_value=0, max_value=100, value=10)
refrigerator = st.number_input("Number of Refrigerators", min_value=0.0, max_value=100.0, value=2.0)
air_conditioner = st.number_input("Number of Air Conditioners", min_value=0.0, max_value=100.0, value=1.0)
television = st.number_input("Number of Televisions", min_value=0.0, max_value=100.0, value=4.0)
monitor = st.number_input("Number of Monitors", min_value=0.0, max_value=100.0, value=1.0)
motor_pump = st.number_input("Number of Motor Pumps", min_value=0, max_value=100, value=0)
month = st.selectbox("Month", list(range(1, 13)))  # Select the month (1-12)
city = st.selectbox("City", data['City'].unique())  # Choose a city
company = st.selectbox("Power Company", data['Company'].unique())  # Choose a power company
monthly_hours = st.number_input("Monthly Usage Hours", min_value=0.0, max_value=1000.0, value=400.0)
tariff_rate = st.number_input("Tariff Rate (₹ per unit)", min_value=0.0, max_value=100.0, value=8.5)

# Prepare the user input for prediction
user_input = {
    'Fan': fan,
    'Refrigerator': refrigerator,
    'AirConditioner': air_conditioner,
    'Television': television,
    'Monitor': monitor,
    'MotorPump': motor_pump,
    'Month': month,
    'City': city,
    'Company': company,
    'MonthlyHours': monthly_hours,
    'TariffRate': tariff_rate
}

# Predict the electricity bill
if st.button("Predict Electricity Bill"):
    input_df = pd.DataFrame([user_input])  # Convert user input to DataFrame
    input_preprocessed = preprocessor.transform(input_df)  # Preprocess user inputs
    predicted_bill = selected_model.predict(input_preprocessed)  # Predict using selected model
    st.success(f"The predicted monthly electricity bill using {algorithm} is ₹{predicted_bill[0]:.2f}")

# Footer
st.markdown('<div class="footer">2024 Electricity Bill Prediction App | Developed by Ekansh Singh</div>', unsafe_allow_html=True)
