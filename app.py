import os
import uuid

import streamlit as st

from analyze import analyze
from simulate import simulate
from translate import Translator

# Page setup
st.set_page_config(
    page_title="MSYFish",
    page_icon="🎣"
)

st.title("MSYFish Model")

# Set up session state
if "id" not in st.session_state:
    if "id" in st.query_params:
        st.session_state.id = st.query_params["id"]
    else:
        st.session_state.id = str(uuid.uuid4())
try:
    os.mkdir(f"simulations/{st.session_state.id}")
except FileExistsError:
    pass

if "running" not in st.session_state:
    st.session_state.running = False

if "language" not in st.session_state:
    st.session_state.language = "en"

if "names" not in st.session_state:
    st.session_state.names = "scientific"

if "plot" not in st.session_state:
    st.session_state.plot = ""

if "fishingDat" not in st.session_state:
    st.session_state.fishingDat = ""

if "popDat" not in st.session_state:
    st.session_state.popDat = ""

if "mode" not in st.session_state:
    st.session_state.mode = "simulate"

# Translator
translator = Translator(st.session_state.language)
t = translator.translate
opt = translator.option

with st.sidebar:
    # Language select
    languages = ["en", "es"]
    language = st.selectbox(
        label=t("language"),
        options=languages,
        index=languages.index(st.session_state.language),
        disabled=st.session_state.running
    )
    # If language changed, rerun page
    if st.session_state.language != language:
        st.session_state.language = language
        st.rerun()

    translator.set_lang(st.session_state.language)
    t = translator.translate
    opt = translator.option

    # Species name select
    name_options = ["scientific", "common"]
    names = st.selectbox(
        label=t("spesies_names"),
        options=name_options,
        index=name_options.index(st.session_state.names),
        format_func=lambda x: opt(x),
        disabled=st.session_state.running
    )
    # If name type changed, rerun page
    if st.session_state.names != names:
        st.session_state.names = names
        st.rerun()

    # Mode select
    mode_options = ["simulate", "analyze"]
    mode = st.selectbox(
        label=t("sim_mode"),
        options=mode_options,
        index=mode_options.index(st.session_state.mode),
        format_func=lambda x: opt(x),
        disabled=st.session_state.running
    )
    # If mode changed, rerun page
    if st.session_state.mode != mode:
        st.session_state.mode = mode
        st.rerun()

    # Link to session
    st.write(
        t("session_link", link=f"{st.context.url}?id={st.session_state.id}")
    )

match st.session_state.mode:
    case "simulate":
        simulate()
    case "analyze":
        analyze()
