import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager
import xgboost
from xgboost import XGBClassifier

# 加载保存的随机森林模型
model = joblib.load('xgb_model.pkl')

# 特征范围定义（根据新提供的变量列表）
feature_ranges = {
    "Norepinephrine": {
        "type": "categorical",
        "options": [0, 1],
        "default": 0
    },
    "Vasopressin": {
        "type": "categorical",
        "options": [0, 1],
        "default": 0
    },
    "Pneumonia": {
        "type": "categorical",
        "options": [0, 1],
        "default": 0
    },
    "age": {
        "type": "numerical",
        "min": 0.0,
        "max": 120.0,
        "default": 50.0,
        "unit": "years old"
    },
    "heartrate": {
        "type": "numerical",
        "min": 0.0,
        "max": 200.0,
        "default": 80.0,
        "unit": "bpm"
    },
    "SBP": {
        "type": "numerical",
        "min": 0.0,
        "max": 300.0,
        "default": 120.0,
        "unit": "mmHg"
    },
    "WBC": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 10.0,
        "unit": "×10⁹/L"
    },
    "Albumin": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 40.0,
        "unit": "g/L"
    },
    "TotalBilirubin": {
        "type": "numerical",
        "min": 0.0,
        "max": 50.0,
        "default": 1.0,
        "unit": "mg/dL"
    },
    "BUN": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 10.0,
        "unit": "mmol/L"
    },
    "Sodium": {
        "type": "numerical",
        "min": 100.0,
        "max": 180.0,
        "default": 140.0,
        "unit": "mmol/L"
    }
}
# Streamlit 界面
st.header("Enter the following feature values:")
feature_values = []

for feature, properties in feature_ranges.items():
    # 统一首字母大写作为展示名
    display_name = feature.capitalize()

    if properties["type"] == "numerical":
        unit = properties.get("unit", "")
        label = f"{display_name} ({unit}, {properties['min']}–{properties['max']})"
        value = st.number_input(
            label=label,
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"])
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{display_name} (Select 0 or 1)",
            options=properties["options"]
        )
    feature_values.append(value)
    
# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    predicted_proba = model.predict_proba(features)[0, 1]   
    probability = predicted_proba * 100                    

    text = f"Based on feature values, predicted possibility of septic shock is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    try:
        prop = font_manager.FontProperties(
            family='Times New Roman', style='italic', weight='bold', size=16)
        ax.text(0.5, 0.5, text, fontproperties=prop,
                ha='center', va='center', transform=ax.transAxes)
    except:
        ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center',
                style='italic', weight='bold', family='serif', transform=ax.transAxes)
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300, transparent=True)
    st.image("prediction_text.png")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[1])
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values[:, :, 0]
    baseline = float(explainer.expected_value)
    sv = shap_values[0]
    shap_fig = shap.force_plot(baseline, sv,
                               pd.DataFrame([feature_values], columns=feature_ranges.keys()),
                               matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")