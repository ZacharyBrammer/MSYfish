import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="MSYFish",
    page_icon="🎣"
)

# Get species data from spreadsheet, can turn into user uploaded file later
fishdata = pd.read_excel("fish_growth_data2.xlsx")
speciesList = fishdata["species"]

# Load connectivity file
conn_data = pd.read_excel("connectivity.xlsx")
connectivity = conn_data.to_numpy()
connectivity = connectivity[:,1:]

st.title("MSYFish Model")

# Have user select species, get indexes for running model
selectedSpecies: list[str] = st.multiselect(label="Species", options=speciesList)
speciesIndexes: list[int] = []

for species in selectedSpecies:
    speciesIndexes.append(speciesList[speciesList == species].index.values[0].item())

# Get output directory
directory = st.text_input(label="Model Output Directory")

# Integer inputs to the model
stocks = st.number_input(label="Number of stocks", step=1, min_value=1)
niter = st.number_input(label="Iterations to run", step=1, min_value=1)
years = st.number_input(label="Years per simulation", step=1, min_value=1)
initialPop = st.number_input(label="Initial population", step=1, min_value=1)


# Optional enables
st.write("Optional Flags")
enableConn = st.toggle(label="Connectivity")
fishing = st.toggle(label="Fishing")
rotation = st.toggle(label="Rotation")


# Run model
for _ in range(len(speciesIndexes)):
    print(True)
