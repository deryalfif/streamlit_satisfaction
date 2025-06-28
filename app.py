import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ------------------------
# Load data dan model
# ------------------------
df = pd.read_csv("E-commerce-Customer-Behavior.csv")

# Load model dan encoder jika tersedia
try:
    model = joblib.load("satisfied_model.pkl")
    le_gender = joblib.load("gender_encoder.pkl")
    le_membership = joblib.load("membership_encoder.pkl")
    le_satisfaction = joblib.load("satisfaction_encoder.pkl")
    
except:
    model = None
    le_gender = None
    le_membership = None
    le_satisfaction = None
   
# ------------------------
# Sidebar: Menu Navigasi
# ------------------------
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Predict Satisfaction"])

# ------------------------
# MENU 1: DASHBOARD
# ------------------------
if menu == "Dashboard":
    st.title("Dashboard E-commerce Customer Behavior")

    # Sidebar filters
    st.sidebar.header("Filter Data")
    
    # City filter
    city = ['All'] + list(df['City'].dropna().unique())
    selected_city = st.sidebar.selectbox("Pilih City", city)
    
    # Gender filter
    genders = df['Gender'].dropna().unique()
    selected_gender = st.sidebar.multiselect("Pilih Gender", genders, default=list(genders))
    
    # Membership filter
    membership = ['All'] + list(df['Membership Type'].dropna().unique())
    selected_membership = st.sidebar.selectbox("Pilih Membership Type", membership)
    
    # Terapkan filter ke data
    filtered_df = df.copy()

    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['City'] == selected_city]

    if selected_membership != 'All':
        filtered_df = filtered_df[filtered_df['Membership Type'] == selected_membership]

    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_gender)]

    # Tampilkan data
    st.dataframe(filtered_df)
    st.markdown(f"Jumlah data yang ditampilkan: {len(filtered_df)}")
    
    st.header("Visualisasi")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Jumlah Customers per Kota")

        # hitung jumlah per kota dan ubah jadi dataframe
        city_counts = filtered_df['City'].value_counts().reset_index()
        city_counts.columns = ['City', 'Count']

        # buat bar chart
        fig_city = px.bar(city_counts, x='City', y='Count',
                        labels={'City': 'Kota', 'Count': 'Jumlah Customers'},
                        title="Jumlah Customers per Kota")
        st.plotly_chart(fig_city)
        
    with col2:
        st.subheader("Distribusi Membership Type")

        membership_counts = filtered_df['Membership Type'].value_counts().reset_index()
        membership_counts.columns = ['Membership Type', 'Count']

        fig_membership = px.pie(
            membership_counts,
            names='Membership Type',
            values='Count',
            title='Distribusi Membership Pelanggan',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_membership)
 
# ------------------------
# MENU 2: PREDICT Satisfaction
# ------------------------   

# le_gender = None
# le_membership = None
# le_satisfaction = None

elif menu == "Predict Satisfaction":
    st.title("Prediksi Kepuasan Pelanggan")

    if model is None or le_gender is None or le_membership is None or le_satisfaction is None:
        st.error("Model atau LabelEncoder belum tersedia. Silakan latih dan simpan terlebih dahulu.")
    else:
        st.subheader("Masukkan Data Pelanggan untuk Prediksi Kepuasan")
    
        membership = st.selectbox("Pilih Membership", le_membership.classes_)
        discount_applied = st.number_input("Discount Applied", min_value=0, max_value=100, value=0)
        days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=0)
        items_purchased = st.number_input("Items Purchased", min_value=0, value=0)
        

        if st.button("Prediksi Kepuasan"):
            encoded_membership = le_membership.transform([membership])[0]
            
            # X_pred disesuaikan dengan fitur yang diminta oleh model
            X_pred = np.array([[discount_applied, days_since_last_purchase, items_purchased , encoded_membership]])
            
            prediction = model.predict(X_pred)[0]
            predicted_label = le_satisfaction.inverse_transform([prediction])[0]

            st.subheader("Hasil Prediksi Kepuasan")
            st.success(f"Prediksi Kepuasan: {predicted_label}")
            
            
    #         df_data[['Discount Applied', 'Days Since Last Purchase',
    #    'Items Purchased', 'Membership Type']]