import streamlit as st
import requests

st.title("Streamlit Frontend")

response = requests.get("http://backend:8000/")
st.write("Backend Response:", response.json())
