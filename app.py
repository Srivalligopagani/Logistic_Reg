import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title
st.title("Social Network Ads Purchase Prediction")

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Encode Gender column
df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0
})

# Features and target
X = df.drop(columns=['User ID', 'Purchased'])
y = df['Purchased']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader(f"Model Accuracy: {accuracy:.2f}")

# User input
st.header("Enter User Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 60, 25)
salary = st.number_input("Estimated Salary", 10000, 200000, 50000)

# Convert gender
gender_value = 1 if gender == "Male" else 0

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame({
        'Gender': [gender_value],
        'Age': [age],
        'EstimatedSalary': [salary]
    })

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.success("User is likely to Purchase")
    else:
        st.error("User is NOT likely to Purchase")