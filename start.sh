#!/bin/bash

source activate msyfish

exec streamlit run app.py --server.headless=true --server.port=8501 --server.address=0.0.0.0