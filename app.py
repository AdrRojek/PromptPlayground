"""Prompt Playground — Streamlit sandbox."""
import streamlit as st

st.set_page_config(page_title="Prompt Playground", layout="wide")
st.title("Prompt Playground")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["openai", "ollama"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)

st.write("Provider:", provider, "temp:", temperature)
