# ============================================================
# 📊 Streamlit App: Exam Mark Prediction
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ================================
# 🔹 Load Dataset
# ================================
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    data['hours'] = data['hours'].fillna(data['hours'].mean())
    return data

dataset = load_data()

# ================================
# 🔹 Train Model
# ================================
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

model = LinearRegression()
model.fit(X, Y)

# ================================
# 🔹 UI Design
# ================================
st.set_page_config(page_title="Exam Mark Predictor", layout="centered")

st.title("📊 Exam Mark Prediction App")
st.write("Predict student exam marks using Multiple Linear Regression")

# ================================
# 🔹 User Inputs
# ================================
st.subheader("Enter Student Details:")

hours = st.slider("📚 Study Hours", 0.0, 12.0, 5.0)
sleep = st.slider("😴 Sleep Hours", 0.0, 12.0, 6.0)
previous = st.slider("📝 Previous Score", 0.0, 100.0, 50.0)

# ================================
# 🔹 Prediction
# ================================
if st.button("🔮 Predict Marks"):
    input_data = np.array([[hours, sleep, previous]])
    prediction = model.predict(input_data)

    st.success(f"🎯 Predicted Exam Mark: {prediction[0]:.2f}")

# ================================
# 🔹 Show Dataset (Optional)
# ================================
if st.checkbox("Show Dataset"):
    st.write(dataset)