import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
# 需要与模型训练时的数据预处理一致
binary_columns = [
    "HadStroke", "PneumoVaxEver", "HadArthritis", "HadKidneyDisease",
    "DeafOrHardOfHearing", "HadHeartAttack", "HadAngina", "ChestScan",
    "DifficultyWalking", "HadDiabetes", "HadCOPD"
]
categorical_columns = ["GeneralHealth", "SmokerStatus", "AgeCategory", "RemovedTeeth"]
numerical_columns = ["PhysicalHealthDays", "MentalHealthDays", "BMI", "HeightInMeters", "WeightInKilograms"]
# One-Hot Coding of Categorical Variables
# Assume the possible values of the categorical variable during training (obtained from the data set)
general_health_values = ["Poor", "Fair", "Good", "VeryGood", "Excellent"]
smoker_status_values = ["Current smoker - now smokes every day", "Current smoker - now smokes some days", "Former smoker", "Never smoked"]
age_category_values = ["Age 80 or older", "Age 75 to 79", "Age 70 to 74", "Age 65 to 69", "Age 60 to 64", "Age 55 to 59", "Age 50 to 54", "Age 45 to 49", "Age 40 to 44", "Age 35 to 39","Age 30 to 34","Age 25 to 29","Age 18 to 24",]
removed_teeth_values = ["1 to 5", "6 or more, but not all", "None of them", "All"]
# 加载模型
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "rf_42features_model.pkl")
feature_path = os.path.join(current_dir, "rf_42features_columns.pkl")
model = joblib.load(model_path)
feature_columns = joblib.load(feature_path)
# 定义干预措施
interventions = {
    "Quit smoking": lambda input_data: {**input_data, "SmokerStatus": "No"},
    "Lose weight": lambda input_data: {**input_data, "BMI": max(18.5, input_data["BMI"] - 5)},
    "Improve mental health": lambda input_data: {**input_data, "MentalHealthDays": max(0, input_data["MentalHealthDays"] - 5)},
    "Reduce physical health problems": lambda input_data: {**input_data, "PhysicalHealthDays": max(0, input_data["PhysicalHealthDays"] - 5)},
    "Increase physical activity": lambda input_data: {**input_data, "DifficultyWalking": "No"},
    "Improve general health": lambda input_data: {**input_data, "GeneralHealth": "Excellent"},
    "Improve diet and oral health": lambda input_data: {**input_data, "RemovedTeeth": "None of them"},
    "Manage diabetes effectively": lambda input_data: {**input_data, "HadDiabetes": "No"},
    "Reduce heart-related risk factors (Stroke/Angina/ChestScan)": lambda input_data: {
        **input_data,
        "HadStroke": "No",
        #"HadHeartAttack": "No",
        "HadAngina": "No",
        "ChestScan": "No",
    },
    "Improve respiratory health": lambda input_data: {**input_data, "HadCOPD": "No"},
    "Enhance kidney health": lambda input_data: {**input_data, "HadKidneyDisease": "No"},
    "Address arthritis issues": lambda input_data: {**input_data, "HadArthritis": "No"},
    "Assist hearing health": lambda input_data: {**input_data, "DeafOrHardOfHearing": "No"},
    "Encourage vaccination": lambda input_data: {**input_data, "PneumoVaxEver": "Yes"},
    "Encourage regular health check-ups": lambda input_data: {**input_data, "ChestScan": "No"},
}

# 定义数据预处理函数
# Data preprocessing function
def preprocess_input(user_input):
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform([[user_input[col] for col in numerical_columns]])[0]
    numerical_input = dict(zip(numerical_columns, user_input_scaled))

    one_hot_encoded_input = {}
    for value in general_health_values:
        one_hot_encoded_input[f"GeneralHealth_{value}"] = 1 if user_input["GeneralHealth"] == value else 0
    for value in smoker_status_values:
        one_hot_encoded_input[f"SmokerStatus_{value}"] = 1 if user_input["SmokerStatus"] == value else 0
    for value in age_category_values:
        one_hot_encoded_input[f"AgeCategory_{value}"] = 1 if user_input["AgeCategory"] == value else 0
    for value in removed_teeth_values:
        one_hot_encoded_input[f"RemovedTeeth_{value}"] = 1 if user_input["RemovedTeeth"] == value else 0

    def binary_encoder(value):
        return 1 if value == "Yes" else 0

    binary_encoded_input = {col: binary_encoder(user_input[col]) for col in binary_columns}

    all_features = {**numerical_input, **one_hot_encoded_input, **binary_encoded_input}
    input_features = [all_features[col] if col in all_features else 0 for col in feature_columns]
    return np.array(input_features).reshape(1, -1)

# 定义预测函数
def predict(input_data):
    preprocessed_data = preprocess_input(input_data)
    risk_prob = model.predict_proba(preprocessed_data)[:, 1][0]
    risk_class = model.predict(preprocessed_data)[0]
    return risk_prob, risk_class

# 定义 Streamlit 界面
st.title("Heart Disease Risk Prediction and Interventions")

# 获取用户输入
st.header("Patient Information")
user_input = {
    "PhysicalHealthDays": st.slider("Physical Unhealthy Days (last 30 days)", 0, 30, 10),
    "MentalHealthDays": st.slider("Mental Unhealthy Days (last 30 days)", 0, 30, 5),
    "BMI": st.number_input("Body Mass Index (BMI)", 10.0, 50.0, 25.0),
    "HeightInMeters": st.number_input("Height (in meters)", 1.0, 2.5, 1.75),
    "WeightInKilograms": st.number_input("Weight (in kilograms)", 30.0, 200.0, 70.0),
    "GeneralHealth": st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"]),
    "SmokerStatus": st.selectbox("Smoker Status", ["Yes", "No"]),
    "AgeCategory": st.selectbox("Age Category", ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59", "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 or older"]),
    "RemovedTeeth": st.selectbox("Removed Teeth", ["None of them", "1 to 5", "6 or more but not all", "All of them"]),
    "HadStroke": st.selectbox("Had Stroke?", ["Yes", "No"]),
    "PneumoVaxEver": st.selectbox("Had Pneumonia Vaccine?", ["Yes", "No"]),
    "HadArthritis": st.selectbox("Had Arthritis?", ["Yes", "No"]),
    "HadKidneyDisease": st.selectbox("Had Kidney Disease?", ["Yes", "No"]),
    "DeafOrHardOfHearing": st.selectbox("Deaf or Hard of Hearing?", ["Yes", "No"]),
    "HadHeartAttack": st.selectbox("Had Heart Attack?", ["Yes", "No"]),
    "HadAngina": st.selectbox("Had Angina?", ["Yes", "No"]),
    "ChestScan": st.selectbox("Chest Scan Done?", ["Yes", "No"]),
    "DifficultyWalking": st.selectbox("Difficulty Walking?", ["Yes", "No"]),
    "HadDiabetes": st.selectbox("Had Diabetes?", ["Yes", "No"]),
    "HadCOPD": st.selectbox("Had COPD?", ["Yes", "No"])
}

# 预测风险
if st.button("Predict Risk"):
    risk_prob, risk_class = predict(user_input)
    st.write(f"**Heart Disease Risk Probability:** {risk_prob * 100:.2f}%")
    st.write(f"**Risk Level:** {'High Risk' if risk_class == 1 else 'Low Risk'}")

# 显示干预措施效果
st.header("Intervention Effects")
if st.button("Simulate Interventions"):
    results = []
    for name, intervention in interventions.items():
        modified_input = intervention(user_input)
        risk_prob, _ = predict(modified_input)
        results.append((name, risk_prob * 100))

    results_df = pd.DataFrame(results, columns=["Intervention", "Risk Probability"])
    st.table(results_df.sort_values(by="Risk Probability"))
