# MSYfish

***MSYfish*** is an individual-based population model for marine fishes, built on dynamic energy budget theory and metabolic scaling. It is the model described in Woodson et al. (2024). This repository contains the model and a web interface (built with [Streamlit](https://streamlit.io)) for running simulations and exploring the results in a browser.

> C.B. Woodson, S.Y. Litvin, J.R. Schramski, S.B. Joye. (2024) An individual-based model for exploration of population and stock dynamics in marine fishes, *Ecological Modelling* 498, 110842, https://doi.org/10.1016/j.ecolmodel.2024.110842.

<details>
<summary>Abstract</summary>

Many size- or age-structure fisheries models require estimation of fundamental population level parameters such as growth, mortality, and recruitment rates that are notoriously difficult to estimate and can constrain the ability of models for exploring emergent properties in population dynamics. To address some of these issues, we develop a discrete-time individual-based model that integrates both age- and size-based concepts. Individual fish are tracked throughout their lifetime allowing for assessment of age-based concepts, with traits determined by size. This method utilizes individual growth parameters as opposed to population level growth rates and allows for many properties of populations that are normally prescribed to be emergent properties of the model. We demonstrate the utility of the model for reproducing population level parameters such as slope at origin for recruitment curves and intrinsic growth rates. The addition of spatial dynamics where a population is sub-divided into discrete stocks further allows for the assessment of various conservation techniques such as marine protected areas, fishing area rotation, and size limits at the individual level.

Keywords: Agent-based modeling; Sized-based traits; Population dynamics; Stock dynamics; Marine fishes
</details>

---

## Contents

1. [Quick start](#quick-start)
2. [The sidebar](#the-sidebar)
3. [Simulate mode](#simulate-mode)
4. [Simulation parameters](#simulation-parameters)
5. [Reading the results](#reading-the-results)
6. [Analyze mode](#analyze-mode)
7. [Input file formats](#input-file-formats)
8. [Output files](#output-files)
9. [Troubleshooting](#troubleshooting)

Developer documentation (running the site yourself and the repository layout) is at the end of the [full README on GitHub](https://github.com/ZacharyBrammer/MSYfish#running-the-site-yourself).

---

## Quick start

1. Open the site. In the sidebar, **copy and save the "Link to Session" URL**. This allows you to come back to your simulations later.
2. Choose a **Species** from the dropdown.
3. Enter a name for the **Model Output Directory**, for example `baseline`. Your results are grouped under this name.
4. Click **Initialize Simulation** and wait for it to finish. This step can take several minutes.
5. Set **Number of stocks**, **Iterations to run**, and **Years per simulation**. You can also turn on and set any of the optional parameters.
6. Click **Run Simulations**. When the run finishes, the summary statistics and plots appear below the button.
7. To look at past runs, build custom plots, or download data, switch **Mode** to `analyze` in the sidebar.

---

## The sidebar

| Option | What it does |
|---|---|
| **Language** | Sets the site language to English (`en`) or Spanish (`es`). |
| **Species Names** | Shows species by `scientific` or `common` name. |
| **Mode** | `simulate` sets up and runs simulations. `analyze` lets you browse, plot, and download past simulations. |
| **Link to Session** | A bookmarkable link to your personal session (`...?id=<session-id>`). |
| **User Manual** | Opens this manual in a new browser tab. |

The sidebar controls are locked while a simulation is running.

### About sessions

The site has no user accounts. Each visitor gets a random session ID, and all of their simulations are stored on the server under that ID.

- **Bookmark the session link.** If you open the site without it, you get a new, empty session.
- **Anyone who has your link can see your simulations.** Only share it with people you want to have access.
- The same link works on another computer or browser.
- If you bookmark this page and `...?id=...` is not present in the URL, **you will not be able to return to your simulations.**

---

## Simulate mode

Running a simulation has two steps: **initialize**, then **run**.

### Step 1: Choose a species and initialize

| Field | Description |
|---|---|
| **Species** | The species to model. Its growth and life-history parameters come from `fish_growth_data2_COJ.xlsx` (see [Species data](#species-data)). |
| **Model Output Directory** | The folder name for this set of runs. Only letters, numbers, `_` and `-` are allowed. Use different names to keep scenarios apart, for example `no_fishing`, `fish_20pct`, `mpa_rotation`. |
| **Background Resource** *(toggle)* | Lets you override the default food/resource level (see [Background resource](#background-resource)). Most users can leave this off. |

When you click **Initialize Simulation**, the model calibrates itself for the chosen species in two steps:

1. **Estimating Recruitment Rate.** The model searches for the lowest larval survival rate that keeps an unfished population going.
2. **Estimating Maximum Fishing Rate.** Starting from 0%, the model raises the fishing rate in steps of 1% until the population can no longer sustain itself.

When initialization is done, the page shows the **Maximum Calculated Fishing Rate**. This is a useful reference for choosing a fishing rate in the next step.

Initialization is saved with your session. If you return to a species you have already initialized, you can go straight to Step 2. Clicking **Initialize Simulation** again recalibrates the model. You need to do this if you change the background resource.

> **Note:** The calibration is stochastic, so initializing the same species twice can give slightly different maximum fishing rates. If a fishing rate of 0% is displayed, you should re-initalize the species.

### Step 2: Configure and run

Set the [simulation parameters](#simulation-parameters) and click **Run Simulations**. The inputs are locked while the simulations run. When they finish, a "Simulations complete" message appears, and the page reloads and shows the results.

**Every simulation starts with a 100-year spin-up period** with no fishing, temperature forcing, or climatic events, so the population can stabilize. The spin-up is added on top of the years you request, and it is not shown in the plots. "Year 0" in every plot is the end of the spin-up. Downloaded files include this spin-up time.

---

## Simulation parameters

### Required

| Parameter | Description |
|---|---|
| **Number of stocks** | The number of spatially separate sub-populations (stocks) the population is split into. The background resource is divided evenly among stocks. |
| **Iterations to run** | How many independent simulations to run with the same settings. The model is stochastic, so running several iterations and looking at the average is recommended. |
| **Years per simulation** | The number of years to simulate after the 100-year spin-up. |

### Optional

Each optional parameter is turned on with a toggle under **Optional Parameters**.

#### Connectivity

Controls how offspring (recruits) move between stocks. When connectivity is off, recruits are assigned to stocks at random.

The connectivity matrix is *N × N*, where *N* is the number of stocks. **Row *i*, column *j* is the fraction of stock *i*'s reproductive output that recruits into stock *j*.** Each row must sum to 1.

You can type values into the on-screen table or upload a file (see [Connectivity file](#connectivity-file)). If a row does not sum to 1, or the matrix size does not match the number of stocks, a warning appears and connectivity is turned off for that run. A matrix of all zeros also turns connectivity off.

#### Fishing

| Setting | Description |
|---|---|
| **Percentage of population to fish** | The fishing rate (0–100%) applied to every stock each year. Compare it with the Maximum Calculated Fishing Rate. Rates above that value are likely to make the population collapse. |
| **Restrict catch size** *(toggle)* | Only fish within a length range are caught. |
| ↳ **Minimum size to catch (cm)** | The smallest length that can be caught. |
| ↳ **Maximum size to catch (cm)** | The largest length that can be caught. Leave blank for no upper limit. |
| **Rotation** *(toggle)* | Rotational closure, like a rotating marine protected area. One stock at a time is closed to fishing, and the closure moves to the next stock every *rotation rate* years. Fishing in the open stocks is scaled up by *N/(N−1)*, so total fishing effort stays the same. **Requires 2 or more stocks.** |
| ↳ **Rotation rate (years)** | The number of years between rotations. Setting this to the full simulation length (years + 100) makes the closure permanent, so stock 1 becomes a permanent reserve. |

#### Set Temperature

Lets you set the water temperature (°C) for each simulated year. Temperature changes fish production (growth) through a metabolic temperature dependence (Boltzmann–Arrhenius, activation energy 0.65 eV). Warmer water means more production.

Enter values in the table (one row per year, defaulting to 20 °C), or upload a file (see [Temperature file](#temperature-file)). Temperature forcing only applies after the spin-up.

#### Climatic Event Enable

Simulates random mass-mortality events such as heat waves or hypoxia.

| Setting | Description |
|---|---|
| **% Chance of Climatic Event** | The probability (0–100%) that an event happens in any given year. |
| **% Population Lost In Event** | The share of the whole population (0–100%) killed at random when an event happens. |

Events can only happen after the spin-up.

#### Production Scaling

**Percentage of production to offset by** (−30% to +30%) multiplies every fish's production (growth) by this percentage. For example, −10% means every fish grows 10% less efficiently. This is useful for exploring general changes in productivity.

#### Background resource

The amount of food available to the population each year. It is split evenly among stocks and varies a little from year to year. By default it scales with the species' asymptotic body size. Changing it changes the population's carrying capacity. Turning it on in Step 1 means you need to initialize again.

<!-- TODO: add guidance on sensible ranges / units for the background resource value -->

---

## Reading the results

After a run finishes, the page shows the output file name, summary statistics, and plots. **If you ran more than one iteration, the results shown are the average across all iterations in that batch.**

### Summary statistics

| Statistic | Meaning |
|---|---|
| Stock Biomass Average / Standard Deviation | Mean and standard deviation of stock biomass across all stocks and years (after spin-up). |
| Average Catch (MT) / Average Catch (#) | Mean yearly catch per stock by weight and by count. Only shown when fishing is on. |

### Plots

| Plot | Shows |
|---|---|
| **Stock Biomass vs Time** | Biomass of each stock over time. |
| **Population Size vs Time** | Total number of individual fish over time. |
| **Catch Per Stock vs Time** | Yearly catch weight for each stock. *(Fishing only)* |
| **Catch Per Stock** | Mean catch for each stock, with standard deviation error bars. *(Fishing only)* |

The plots are interactive: hover to see values, drag to zoom, double-click to reset, and click legend entries to show or hide stocks. Use the camera icon to save a plot as a PNG.

If a plot title says **"Warning: population crashed during simulation"**, the population went extinct before the end of the run, and the plot stops at that year.

---

## Analyze mode

Switch **Mode** to `analyze` in the sidebar to work with any simulation saved in your session.

1. **Download all simulations (.zip)** downloads every NetCDF file in your session at once.
2. **Folder**: pick an output directory.
3. **Species**: pick a species in that folder.
4. Either:
   - pick a single **Simulation** file, or
   - turn on **Compare all simulations** to average every simulation in that folder for that species.
5. **Download NetCDF file** downloads the selected (or averaged) file.
6. **Dataset Variables**: expand this to see a description of each variable in the file.
7. **Select Variables (Custom Plotting)**: choose one or more variables. They appear in a table that you can download as CSV from the table's toolbar, and in a **Custom Plot** against time. Variables split by stock or size bin show up as one column per stock or bin.
8. The standard plots (the same ones as in simulate mode) are shown below the custom plot.

> **Tip:** "Compare all simulations" averages *every* file in the folder, even ones run with different settings. Use a separate output directory for each scenario so the averages make sense.

---

## Input file formats

Connectivity and temperature data can be uploaded as `.csv`, `.xls`, or `.xlsx`.

### Connectivity file

The first row holds stock labels, the first column holds stock labels, and the rest is the *N × N* matrix. Each row must sum to 1. An example is in [connectivity.xlsx](connectivity.xlsx):

| Stock | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P1 | 0.7 | 0.2 | 0.1 | 0.0 |
| P2 | 0.0 | 0.8 | 0.2 | 0.0 |
| P3 | 0.0 | 0.0 | 0.6 | 0.4 |
| P4 | … | … | … | … |

### Temperature file

Two columns: the year (1, 2, 3, …) and the temperature in °C. The first row must be a header row (for example `year,temperature`), for both CSV and Excel files. After the header, there must be exactly one row for each simulated year.

```
year,temperature
1,20.0
2,20.3
3,20.1
```

### Species data

Species parameters are read from [fish_growth_data2_COJ.xlsx](fish_growth_data2_COJ.xlsx). Each row is a species:

| Column | Meaning |
|---|---|
| `order`, `family`, `scientific` | Taxonomy |
| `common_en`, `common_es` | Common name (English / Spanish) |
| `Lmean`, `Lstd` | Asymptotic length L∞ (cm): mean and SD |
| `Kmean`, `Kstd` | von Bertalanffy growth coefficient K: mean and SD |
| `amean`, `astd`, `bmean`, `bstd` | Length–weight relationship W = aL^b: mean and SD |
| `slope`, `inter` | Regression used to derive *a* from *b* |
| `max age` | Maximum age (years) |

---

## Output files

Each simulation is saved as a NetCDF (`.nc`) file:

```
simulations/<session-id>/<output-directory>/<Genus_species>/
    msy_stocks_<N>_nfish_<F>_mfish_<rate>_rot_<RRR>_<iteration>.nc
    average.nc
```

| Part of name | Meaning |
|---|---|
| `stocks_<N>` | Number of stocks |
| `nfish_<F>` | Number of fished stocks (0 when fishing is off) |
| `mfish_<rate>` | Fishing rate as a fraction (e.g. `0.1000` = 10%) |
| `rot_<RRR>` | Rotation rate in years (`000` = no rotation) |
| `<iteration>` | Run counter |

`average.nc` is the average of the most recent multi-iteration batch, or of the last "Compare all simulations". **It is overwritten each time**, so download it if you want to keep it.

The files include the 100-year spin-up, so time index 100 is "year 0" in the plots. They can be opened with Python (`netCDF4`, `xarray`), R (`ncdf4`), MATLAB, or [Panoply](https://www.giss.nasa.gov/tools/panoply/).

### Variables

| Variable | Description | Dimensions |
|---|---|---|
| `ftime` | Time since model start (years) | time |
| `popsize` | Total population size (# individuals) | time |
| `biomass` | Total biomass (MT) | time |
| `sex_ratio` | Proportion female | time |
| `newgrowth` | New biomass produced (MT) | time |
| `consumption` | Resource consumed (MT) | time |
| `reprod_out` | Reproductive output (# eggs) | time |
| `wfish` | Size of target harvest (kg) | time |
| `stock_size` | Individuals per stock | time × stock |
| `stock_biomass` | Biomass per stock (MT) | time × stock |
| `stock_recruit` | Recruits per stock | time × stock |
| `resource` | Resource available per stock | time × stock |
| `reproduction` | Reproduction (number, MT) | time × 2 |
| `mortality` | Mortality (number, MT) | time × 2 |
| `catch` | Catch per stock (number, MT) | time × stock × 2 |
| `pop_bins`, `biomass_bins`, `catch_bins`, `reprod_bins`, `mort_bins` | Size-binned values (24 log-spaced bins) | time × bin × stock |
| `age_bins` | Number of fish in each age class | time × age × stock |
| `mid_bins` | Midpoint of each size bin (kg) | bin |
| `fish`, `age` | Size and age of every individual | time × individual |
| `fishing_rate`, `reprod_rate`, `rotation_rate` | Run settings | scalar / stock |

---

## Troubleshooting

| Problem | Cause / fix |
|---|---|
| My simulations are gone. | You probably opened the site without your session link. Open your bookmarked link. |
| "Invalid folder name" | The output directory can only contain letters, numbers, `_`, and `-`. |
| "Rows must sum up to 1. Connectivity Disabled" | Check that each row of the connectivity matrix sums to exactly 1. |
| "Connectivity file does not match number of stocks" | The uploaded matrix must be N × N, where N is the number of stocks. |
| "File does not fit dimensions" (temperature) | The file needs a header row, then exactly one row per simulated year, with only the year and temperature columns. |
| Rotation was turned off. | Rotation needs at least 2 stocks. |
| "Population crashed during simulation" | The population went extinct. Try a lower fishing rate, fewer or milder climatic events, or more background resource. |
| Initialization takes a long time. | This is normal, especially for long-lived species. Leave the page open until it finishes. |

---

<!-- developer-docs: everything below this line is hidden from the in-app user manual -->

## Running the site yourself

### With Docker (recommended)

```bash
docker build -t msyfish .
docker run -p 8501:8501 -v "$(pwd)/simulations:/app/simulations" msyfish
```

Then open http://localhost:8501. The `-v` mount keeps simulations on the host when the container is removed.

### With conda

```bash
conda env create -f environment.yml
conda activate msyfish
mkdir -p simulations
streamlit run app.py
```

<!-- TODO: note the hosted URL of the public site here, and who to contact for access/issues -->

---

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point: page setup, sessions, sidebar |
| `simulate_page.py` | Simulate mode UI |
| `analyze.py` | Analyze mode UI and averaging of runs (`average_sims`) |
| `simulator.py` | `Simulator` class: per-species initialization (recruitment and max fishing rate calibration), running, and saving/loading sessions |
| `msy_stocks_rev14_connect.py` | Core individual-based model (`compute_pop_msy`) and NetCDF writer |
| `plotting.py` | Standard result plots |
| `translate.py`, `labels.json` | English/Spanish UI text (`old_labels.json` is the previous, unused version) |
| `fish_growth_data2_COJ.xlsx` | Species parameters used by the site |
| `connectivity.xlsx` | Example connectivity matrix |
| `max_fish_estimate/` | Offline script that batch-estimates max fishing rates for all species |
| `Dockerfile`, `start.sh`, `environment.yml` | Deployment |

## Adding Custom Species
To add a species, add a row with these columns. Parameters can be taken from [FishLife](https://github.com/James-Thorson-NOAA/FishLife) or [rfishbase](https://github.com/ropensci/rfishbase).
