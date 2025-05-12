import json

import streamlit as st

from analyze import analyze
from simulate import simulate

# Page setup
st.set_page_config(
    page_title="MSYFish",
    page_icon="🎣"
)

# Language setup
languages = ["en", "es"]
with open("labels.json", "r") as f:
    labels = json.load(f)
    f.close()

st.title("MSYFish Model")

# Set up session state
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

with st.sidebar:
    language = st.selectbox(label="Language", options=languages, disabled=st.session_state.running)
    # If language changed, rerun page
    if st.session_state.language != language:
        st.session_state.language = language
        st.rerun()
    
    names = st.selectbox(label="Species Names", options=["scientific", "common"], disabled=st.session_state.running)
    # If name type changed, rerun page
    if st.session_state.names != names:
        st.session_state.names = names
        st.rerun()
    
    mode = st.selectbox(label="Mode", options=["simulate", "analyze"], disabled=st.session_state.running)
    # If mode changed, rerun page
    if st.session_state.mode != mode:
        st.session_state.mode = mode
        st.rerun()

match st.session_state.mode:
    case "simulate":
        simulate(labels)
    case "analyze":
        analyze(labels)
