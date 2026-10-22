"""Prompt Playground — Streamlit sandbox for testing prompts."""
import streamlit as st
from src.config import load_config
from src.history import append_entry, make_entry
from src.llm_client import LLMError, chat

st.set_page_config(page_title="Prompt Playground", page_icon="🧪", layout="wide")
config = load_config()

st.title("Prompt Playground")
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["openai", "ollama"],
                            index=0 if config.preferred_provider == "openai" else 1)
    openai_key = st.text_input("OpenAI API key", value=config.openai_api_key or "", type="password")
    ollama_url = st.text_input("Ollama base URL", value=config.ollama_base_url)
    model = st.text_input("Model", value=config.openai_model if provider == "openai" else config.ollama_model)
    temperature = st.slider("Temperature", 0.0, 2.0, float(config.default_temperature), 0.05)
    max_tokens = st.number_input("Max tokens", 16, 8192, int(config.default_max_tokens), 16)

system_prompt = st.text_area("System prompt", "You are a helpful assistant.", height=120)
user_prompt = st.text_area("User prompt", "Explain temperature in LLMs in one short paragraph.", height=160)

if st.button("Run", type="primary", use_container_width=True):
    if not user_prompt.strip():
        st.warning("User prompt is required.")
    else:
        with st.spinner("Calling model..."):
            try:
                result = chat(
                    provider=provider, model=model, system=system_prompt, user=user_prompt,
                    temperature=float(temperature), max_tokens=int(max_tokens),
                    openai_api_key=openai_key or None, ollama_base_url=ollama_url,
                )
            except LLMError as exc:
                st.error(str(exc))
            else:
                st.subheader("Response")
                st.markdown(result.content)
                st.caption(f"{result.model} · temp={result.temperature} · {result.latency_ms} ms")
                append_entry(config.history_path, make_entry(
                    provider=result.provider, model=result.model, temperature=result.temperature,
                    max_tokens=int(max_tokens), system_prompt=system_prompt, user_prompt=user_prompt,
                    response=result.content, latency_ms=result.latency_ms, meta=result.raw_meta,
                ))
                st.toast("Saved to history")
