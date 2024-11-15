from functools import cache
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from loguru import logger

import os
import sys

project_root = '/mnt/c/Users/a884470/prj/genai-blueprint-main'  # Change this if needed
sys.path.append(os.path.join(project_root, 'python'))

from ai_core.cache import LlmCache
from ai_core.llm import LlmFactory, set_cache
from config import get_config_str, set_config_str

st.set_page_config(
    page_title="GenAI Lab and Practicum",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(verbose=True)

logger.info("Start Webapp...")

logo_an = str(Path.cwd() / "static" / "AcademieNumerique_Colour_RGB-150x150.jpg")
logo_eviden = str(Path.cwd() / "static/eviden-logo-white.png")


st.sidebar.success("Select a demo above.")

title_col1, title_col2, title_col3 = st.columns([3, 1, 1])
title_col2.image(logo_eviden, width=120)
title_col2.image(logo_an, width=120)
title_col1.markdown(
    """
    ## Demos and practicum floor<br>
    **👈 Select one from the sidebar** """,
    unsafe_allow_html=True,
)

@cache
def config_sidebar():
    with st.sidebar:
        with st.expander("LLM Configuration", expanded=True):
            current_llm = get_config_str("llm", "default_model")
            index = LlmFactory().known_items().index(current_llm)
            llm = st.selectbox(
                "default", LlmFactory().known_items(), index=index, key="select_llm"
            )
            set_config_str("llm", "default_model", str(llm))

            set_debug(
                st.checkbox(
                    label="Debug",
                    value=False,
                    help="LangChain debug mode",
                )
            )
            set_verbose(
                st.checkbox(
                    label="Verbose",
                    value=True,
                    help="LangChain verbose mode",
                )
            )

            set_cache(
                LlmCache.from_value(
                    st.selectbox("Cache", ["memory", "sqlite"], index=1)
                )
            )

            if "LUNARY_APP_ID" in os.environ:
                if st.checkbox(
                    label="Use Lunary.ai for monitoring", value=False, disabled=True
                ):
                    set_config_str("monitoring", "default", "lunary")
            if "LANGCHAIN_API_KEY" in os.environ:
                if st.checkbox(label="Use LangSmith for monitoring", value=True):
                    set_config_str("monitoring", "default", "langsmith")
                    os.environ["LANGCHAIN_TRACING_V2"] = "true"
                    os.environ["LANGCHAIN_PROJECT"] = get_config_str(
                        "monitoring", "project"
                    )
                    os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "1.0"

                else:
                    os.environ["LANGCHAIN_TRACING_V2"] = "false"
                    set_config_str("monitoring", "default", "none")


config_sidebar()
