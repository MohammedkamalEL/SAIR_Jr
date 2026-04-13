import streamlit as st
import pandas as pd
import joblib
import os

# 1. إعداد المسارات بشكل ذكي (لنتفادى أخطاء FileNotFoundError)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# تأكد أن اسم الملف هنا مطابق لاسم الملف الذي حفظته في الـ Notebook
MODEL_PATH = os.path.join(BASE_DIR, 'production_models', 'full_insurance_pipeline.pkl')

# 2. تحميل الـ Pipeline الشامل (المعالج + الموديل)
# ملاحظة: لم نعد بحاجة لتحميل model_columns لأنها مخزنة داخل الـ Pipeline
model_pipeline = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Medical Insurance Predictor", page_icon="🏥")

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("---")
st.write("Enter patient details to get an instant insurance cost prediction.")

# 3. واجهة المدخلات (User Inputs)
# نستخدم النصوص مباشرة (Yes/No, Male/Female) لأن الـ Pipeline سيتعامل معها
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoker?", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# 4. التوقع (Prediction)
if st.button("Predict Insurance Cost"):
    # إنشاء DataFrame بالبيانات الخام تماماً كما في ملف الـ CSV
    input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    # الـ Pipeline سيقوم بالتحويل (Encoding) والقياس (Scaling) ثم التوقع تلقائياً
    prediction = model_pipeline.predict(input_df)
    
    st.markdown("---")
    st.success(f"### 💰 Estimated Cost: ${prediction[0]:,.2f}")
    
    # إضافة لمسة احترافية: تنبيه إذا كان الشخص مدخناً
    if smoker == "yes":
        st.warning("⚠️ High charges are mainly due to smoking status.")
        

# run comand app
# streamlit run "SAIR_Jr/1_Regression/Regression Capstone Projects/Mohammed_Kamal/app.py"
