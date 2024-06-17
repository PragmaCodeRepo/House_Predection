import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit app
st.title("House Price Prediction")

# Function to load and preprocess data
def load_data():
    data = {
        'SquareFeet': [1500, 2000, 2500, 3000, 3500, 4000, 4500],
        'Price': [300000, 400000, 500000, 600000, 700000, 800000, 900000]
    }
    df = pd.DataFrame(data)
    return df

# Load data
data = load_data()

# Display the data in the Streamlit app
st.write("### House Prices Dataset")
st.write(data)

# Split the data into features and target variable
X = data[['SquareFeet']]
y = data['Price']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Display the model's performance
st.write("### Model Performance")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

# Interactive prediction
st.write("### Predict House Price")
square_feet = st.slider("Square Feet", min_value=1000, max_value=5000, value=2000, step=100)
predicted_price = model.predict(np.array([[square_feet]]))[0]
st.write(f"Predicted Price for {square_feet} square feet: ${predicted_price:.2f}")
