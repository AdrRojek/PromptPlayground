"""Prompt Playground — Streamlit sandbox for testing prompts."""

from __future__ import annotations

import streamlit as st

from src.compare import CompareSettings, run_compare
from src.config import OPENAI_MODELS, load_config
from src.history import append_entry, load_entries, make_entry
from src.llm_client import LLMError, chat, list_ollama_models

st.set_page_config(
    page_title="Prompt Playground",
    page_icon="🧪",
    layout="wide",
)

config = load_config()


def _provider_models(provider: str, ollama_url: str) -> list[str]:
    if provider == "openai":
        return OPENAI_MODELS
    models = list_ollama_models(ollama_url)
    if config.ollama_model and config.ollama_model not in models:
        models = [config.ollama_model, *models]
    return models or [config.ollama_model]


def _render_result(content: str, latency_ms: int, model: str, temperature: float) -> None:
    st.markdown(f"**Model:** `{model}` · **temp:** `{temperature}` · **latency:** `{latency_ms} ms`")
    st.markdown(content)


def main() -> None:
    st.title("Prompt Playground")
    st.caption("Sandbox for system/user prompts, temperature tuning, A/B comparison, and run history.")

    with st.sidebar:
        st.header("Settings")
        default_provider = config.preferred_provider
        provider = st.selectbox(
            "Provider",
            options=["openai", "ollama"],
            index=0 if default_provider == "openai" else 1,
        )

        openai_key = config.openai_api_key or ""
        if provider == "openai":
            openai_key = st.text_input(
                "OpenAI API key",
                value=openai_key,
                type="password",
                help="Defaults to OPENAI_API_KEY from the environment.",
            )

        ollama_url = st.text_input("Ollama base URL", value=config.ollama_base_url)
        models = _provider_models(provider, ollama_url)
        default_model = config.openai_model if provider == "openai" else config.ollama_model
        model_index = models.index(default_model) if default_model in models else 0
        model = st.selectbox("Model", options=models, index=model_index)

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(config.default_temperature),
            step=0.05,
        )
        max_tokens = st.number_input(
            "Max tokens",
            min_value=16,
            max_value=8192,
            value=int(config.default_max_tokens),
            step=16,
        )

        mode = st.radio("Mode", options=["Single run", "Compare A/B"], horizontal=True)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant.",
        height=120,
    )
    user_prompt = st.text_area(
        "User prompt",
        value="Explain temperature in LLMs in one short paragraph.",
        height=160,
    )

    if mode == "Single run":
        if st.button("Run", type="primary", use_container_width=True):
            if not user_prompt.strip():
                st.warning("User prompt is required.")
            else:
                with st.spinner("Calling model..."):
                    try:
                        result = chat(
                            provider=provider,  # type: ignore[arg-type]
                            model=model,
                            system=system_prompt,
                            user=user_prompt,
                            temperature=float(temperature),
                            max_tokens=int(max_tokens),
                            openai_api_key=openai_key or None,
                            ollama_base_url=ollama_url,
                        )
                    except LLMError as exc:
                        st.error(str(exc))
                    else:
                        st.subheader("Response")
                        _render_result(
                            result.content,
                            result.latency_ms,
                            result.model,
                            result.temperature,
                        )
                        entry = make_entry(
                            provider=result.provider,
                            model=result.model,
                            temperature=result.temperature,
                            max_tokens=int(max_tokens),
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            response=result.content,
                            latency_ms=result.latency_ms,
                            mode="single",
                            meta=result.raw_meta,
                        )
                        append_entry(config.history_path, entry)
                        st.toast("Saved to history")
    else:
        st.subheader("Compare settings")
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("**Variant A**")
            a_provider = st.selectbox("Provider A", ["openai", "ollama"], key="a_provider")
            a_models = _provider_models(a_provider, ollama_url)
            a_model = st.selectbox("Model A", a_models, key="a_model")
            a_temp = st.slider("Temperature A", 0.0, 2.0, 0.2, 0.05, key="a_temp")
            a_tokens = st.number_input("Max tokens A", 16, 8192, int(max_tokens), 16, key="a_tokens")
        with right_col:
            st.markdown("**Variant B**")
            b_provider = st.selectbox("Provider B", ["openai", "ollama"], index=1, key="b_provider")
            b_models = _provider_models(b_provider, ollama_url)
            b_model = st.selectbox("Model B", b_models, key="b_model")
            b_temp = st.slider("Temperature B", 0.0, 2.0, 1.0, 0.05, key="b_temp")
            b_tokens = st.number_input("Max tokens B", 16, 8192, int(max_tokens), 16, key="b_tokens")

        if st.button("Compare", type="primary", use_container_width=True):
            if not user_prompt.strip():
                st.warning("User prompt is required.")
            else:
                with st.spinner("Running both variants..."):
                    left, right = run_compare(
                        system=system_prompt,
                        user=user_prompt,
                        left=CompareSettings(
                            label="A",
                            provider=a_provider,  # type: ignore[arg-type]
                            model=a_model,
                            temperature=float(a_temp),
                            max_tokens=int(a_tokens),
                        ),
                        right=CompareSettings(
                            label="B",
                            provider=b_provider,  # type: ignore[arg-type]
                            model=b_model,
                            temperature=float(b_temp),
                            max_tokens=int(b_tokens),
                        ),
                        openai_api_key=openai_key or config.openai_api_key,
                        ollama_base_url=ollama_url,
                    )

                out_a, out_b = st.columns(2)
                for column, outcome in ((out_a, left), (out_b, right)):
                    with column:
                        st.markdown(f"### Variant {outcome.settings.label}")
                        if outcome.error:
                            st.error(outcome.error)
                            continue
                        assert outcome.result is not None
                        _render_result(
                            outcome.result.content,
                            outcome.result.latency_ms,
                            outcome.result.model,
                            outcome.result.temperature,
                        )
                        entry = make_entry(
                            provider=outcome.result.provider,
                            model=outcome.result.model,
                            temperature=outcome.result.temperature,
                            max_tokens=outcome.settings.max_tokens,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            response=outcome.result.content,
                            latency_ms=outcome.result.latency_ms,
                            mode="compare",
                            label=outcome.settings.label,
                            meta=outcome.result.raw_meta,
                        )
                        append_entry(config.history_path, entry)
                st.toast("Compare runs saved to history")

    st.divider()
    st.subheader("History")
    filter_provider = st.selectbox(
        "Filter by provider",
        options=["all", "openai", "ollama"],
        index=0,
    )
    entries = load_entries(
        config.history_path,
        provider=None if filter_provider == "all" else filter_provider,
        limit=50,
    )
    if not entries:
        st.info("No runs logged yet.")
    else:
        for entry in entries:
            title = (
                f"{entry.timestamp} · {entry.provider}/{entry.model} · "
                f"temp={entry.temperature} · {entry.latency_ms}ms"
            )
            if entry.label:
                title += f" · {entry.mode}:{entry.label}"
            with st.expander(title):
                st.markdown("**System**")
                st.code(entry.system_prompt or "(empty)")
                st.markdown("**User**")
                st.code(entry.user_prompt)
                st.markdown("**Response**")
                st.markdown(entry.response)


if __name__ == "__main__":
    main()
