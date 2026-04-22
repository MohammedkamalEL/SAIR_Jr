import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import datetime # استيراد مكتبة الوقت

# Configuration
st.set_page_config(page_title="Albacete Gas Hub", layout="wide", page_icon="⛽")

# Load Assets
@st.cache_resource
def load_data_and_model():
    # تأكد من أن المسار يطابق مكان حفظ الموديل في الـ Notebook
    model = joblib.load('production_models/gas_price_model.pkl')
    stations = pd.read_csv('data/gasolineras_ab.csv')
    prices = pd.read_csv('data/precios_gasolineras.csv')
    df = pd.merge(prices, stations, on='id_estacion').dropna(subset=['precio_gasolina_95'])
    return model, df

model, df = load_data_and_model()

# Header
st.title("⛽ Gas Station Intelligence System - Albacete")
st.markdown("---")

# Main Layout
tab1, tab2 = st.tabs(["📊 Market Analysis", "🔮 Price Predictor"])

with tab1:
    st.header("Exploratory Insights")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Geographic Price Heatmap")
        fig = px.scatter_mapbox(df, lat="latitud", lon="longitud", color="precio_gasolina_95",
                                size="precio_gasolina_95", zoom=11, mapbox_style="carto-positron",
                                height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Cheapest Brands")
        avg_brand = df.groupby('rotulo')['precio_gasolina_95'].mean().sort_values().head(10)
        st.bar_chart(avg_brand)

with tab2:
    st.header("Instant Price Prediction")
    st.write("Enter coordinates and brand to estimate current price.")
    
    with st.form("pred_form"):
        col_a, col_b = st.columns(2)
        brand = col_a.selectbox("Station Brand", df['rotulo'].unique())
        lat = col_b.number_input("Latitude", value=38.99, format="%.6f")
        lon = col_a.number_input("Longitude", value=-1.85, format="%.6f")
        
        if st.form_submit_button("Run AI Analysis"):
     
            current_month = datetime.datetime.now().month
            
            input_data = pd.DataFrame([[lat, lon, brand, current_month]], 
                                     columns=['latitud', 'longitud', 'rotulo', 'month'])
            
            price = model.predict(input_data)[0]
            
            st.metric("Estimated Price", f"{price:.3f} €")
            
            st.caption(f"Prediction based on current month: {current_month}")