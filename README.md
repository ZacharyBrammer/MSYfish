# msy_fisheries

***MSYfish*** is the individual based population model based on principles of dynamic energy budget theory and metabolic scaling used in Woodson (submitted) to assess the maximum sustainable yield of global fisheries.

To run model, set input and output directories in **calc_msy_parallel_serve.py**. This file is the executable and will call other components. Inputs for an individual species can be added to the input file 'fish_growth_data2' or a new file can be created using FishLife or rfishbase.
