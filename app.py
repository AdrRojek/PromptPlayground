"""Prompt Playground — Streamlit sandbox."""
import streamlit as st

st.set_page_config(page_title="Prompt Playground", layout="wide")
st.title("Prompt Playground")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["openai", "ollama"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.number_input("Max tokens", 16, 8192, 1024, 16)

system_prompt = st.text_area("System prompt", "You are a helpful assistant.", height=120)
user_prompt = st.text_area("User prompt", height=160)
st.button("Run", type="primary")
