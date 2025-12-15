import streamlit as st
import pandas as pd
import os

st.title("✅ STREAMLIT + CSV TEST")

st.write("📁 File di repository:")
st.write(os.listdir())

df = pd.read_csv("synthetic_customers_cleaned (1).csv")
st.success("CSV berhasil dibaca!")
st.write(df.head())
